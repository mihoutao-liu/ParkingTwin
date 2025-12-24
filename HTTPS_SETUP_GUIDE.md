# HTTPS 配置指南

## 问题说明

GitHub Pages 使用 HTTPS 协议，而浏览器的安全策略不允许 HTTPS 页面加载 HTTP 资源（Mixed Content 混合内容）。

当前您的服务器 URL：`http://118.24.188.137:9000/lxh/mesh.ply` 使用的是 HTTP 协议，因此无法在 GitHub Pages 上加载。

## 解决方案

### 方案 1：为您的服务器配置 HTTPS（推荐）⭐

如果您使用的是 MinIO 或类似的对象存储服务，需要配置 SSL 证书。

#### 步骤 A：使用 Nginx 反向代理 + Let's Encrypt

1. **安装 Nginx**
```bash
sudo apt update
sudo apt install nginx
```

2. **配置域名（如果有）**
   - 将域名 DNS 解析到 `118.24.188.137`
   - 假设域名为 `storage.yourdomain.com`

3. **安装 Certbot（Let's Encrypt 客户端）**
```bash
sudo apt install certbot python3-certbot-nginx
```

4. **申请 SSL 证书**
```bash
sudo certbot --nginx -d storage.yourdomain.com
```

5. **配置 Nginx 反向代理**

编辑 `/etc/nginx/sites-available/minio`：
```nginx
server {
    listen 443 ssl http2;
    server_name storage.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/storage.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/storage.yourdomain.com/privkey.pem;
    
    # SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # CORS 配置
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' 'GET, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' '*' always;
    
    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 大文件支持
        client_max_body_size 500M;
    }
}

server {
    listen 80;
    server_name storage.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

6. **启用配置并重启 Nginx**
```bash
sudo ln -s /etc/nginx/sites-available/minio /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

7. **更新代码中的 URL**
```javascript
modelURL = 'https://storage.yourdomain.com/lxh/mesh.ply';
```

#### 步骤 B：直接为 MinIO 配置 HTTPS

如果您不想使用 Nginx，也可以直接为 MinIO 配置 SSL：

1. 将证书文件放到 MinIO 配置目录：
```bash
mkdir -p ~/.minio/certs
cp fullchain.pem ~/.minio/certs/public.crt
cp privkey.pem ~/.minio/certs/private.key
```

2. 重启 MinIO
```bash
sudo systemctl restart minio
```

3. MinIO 会自动在 9000 端口启用 HTTPS

---

### 方案 2：使用 Cloudflare Tunnel（零配置 HTTPS）

Cloudflare Tunnel 可以为您的服务器提供免费的 HTTPS，无需配置证书。

#### 步骤

1. **安装 cloudflared**
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

2. **登录 Cloudflare**
```bash
cloudflared tunnel login
```

3. **创建 Tunnel**
```bash
cloudflared tunnel create parkingtwin-storage
```

4. **配置 Tunnel**

创建配置文件 `~/.cloudflared/config.yml`：
```yaml
tunnel: <TUNNEL-ID>
credentials-file: /home/user/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: storage.yourdomain.com
    service: http://localhost:9000
  - service: http_status:404
```

5. **运行 Tunnel**
```bash
cloudflared tunnel run parkingtwin-storage
```

6. **在 Cloudflare DNS 中添加记录**
   - 类型：CNAME
   - 名称：storage
   - 内容：<TUNNEL-ID>.cfargotunnel.com

现在您可以通过 `https://storage.yourdomain.com/lxh/mesh.ply` 访问文件！

---

### 方案 3：使用免费云存储（已支持 HTTPS）

#### Cloudflare R2
- 自动支持 HTTPS
- 免费 10GB 存储
- 零流量费用

#### 阿里云 OSS
- 自动支持 HTTPS
- 可以使用 CDN 加速
- 价格便宜

**参考之前提供的配置指南**

---

### 方案 4：GitHub Pages 本地预览（临时方案）

当前代码已经添加了环境检测：
- 在 GitHub Pages（HTTPS）上：显示提示信息，引导用户本地查看
- 在本地（HTTP）上：正常加载模型

用户可以：
1. 克隆项目到本地
2. 将 `mesh.ply` 放到 `assets/` 文件夹
3. 直接打开 `index.html` 文件查看

---

## 推荐方案对比

| 方案 | 难度 | 成本 | 访问速度 | 推荐度 |
|------|------|------|----------|--------|
| Nginx + Let's Encrypt | 中 | 免费 | 快（自己服务器） | ⭐⭐⭐⭐ |
| Cloudflare Tunnel | 低 | 免费 | 快（CF CDN） | ⭐⭐⭐⭐⭐ |
| Cloudflare R2 | 低 | 免费 | 快（全球 CDN） | ⭐⭐⭐⭐⭐ |
| 本地预览 | 极低 | 免费 | 最快（本地） | ⭐⭐⭐ |

---

## 快速检查清单

配置完成后，使用以下命令检查：

```bash
# 检查 SSL 证书
curl -I https://your-domain.com/lxh/mesh.ply

# 检查 CORS 配置
curl -I -H "Origin: https://mihoutao-liu.github.io" https://your-domain.com/lxh/mesh.ply
```

应该看到：
- `HTTP/2 200` 或 `HTTP/1.1 200`
- `Access-Control-Allow-Origin: *`

---

## 需要帮助？

如果在配置过程中遇到问题，请检查：
1. 防火墙是否开放了 443 端口
2. DNS 解析是否正确
3. SSL 证书是否有效
4. CORS 配置是否正确

相关文档：
- Let's Encrypt: https://letsencrypt.org/
- Cloudflare Tunnel: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- Nginx HTTPS: https://nginx.org/en/docs/http/configuring_https_servers.html

