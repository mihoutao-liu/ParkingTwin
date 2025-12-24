# 服务器 CORS 跨域配置指南

## 问题说明
当从 `https://mihoutao-liu.github.io` 访问 `http://118.24.188.137:9000/lxh/mesh.ply` 时，会遇到 CORS 跨域限制。

需要在服务器端配置允许跨域访问。

---

## 方案 1：MinIO CORS 配置（推荐）

### 使用 MinIO Client (mc) 配置

#### 1. 安装 MinIO Client

**Linux/macOS:**
```bash
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

**Windows:**
```powershell
# 下载 mc.exe 并添加到 PATH
```

#### 2. 配置 MinIO 连接

```bash
mc alias set myminio http://118.24.188.137:9000 YOUR_ACCESS_KEY YOUR_SECRET_KEY
```

#### 3. 设置 Bucket CORS 规则

创建 CORS 配置文件 `cors.json`:

```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag", "Content-Length"],
      "MaxAgeSeconds": 3600
    }
  ]
}
```

#### 4. 应用 CORS 配置

```bash
mc anonymous set-json cors.json myminio/lxh
```

#### 5. 验证配置

```bash
mc anonymous get-json myminio/lxh
```

### 使用 Python SDK 配置

```python
from minio import Minio

# 连接 MinIO
client = Minio(
    "118.24.188.137:9000",
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
    secure=False
)

# 设置 CORS
cors_config = {
    "CORSRules": [
        {
            "AllowedOrigin": ["*"],
            "AllowedMethod": ["GET", "HEAD"],
            "AllowedHeader": ["*"],
            "ExposeHeader": ["ETag"],
            "MaxAgeSeconds": 3600
        }
    ]
}

client.set_bucket_cors("lxh", cors_config)
```

---

## 方案 2：使用 Nginx 反向代理

如果您的服务器前面有 Nginx，可以在 Nginx 配置 CORS。

### Nginx 配置示例

```nginx
server {
    listen 80;
    server_name 118.24.188.137;

    location /lxh/ {
        # 添加 CORS 响应头
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, HEAD, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' '*' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length, Content-Range, ETag' always;
        add_header 'Access-Control-Max-Age' 3600 always;

        # 处理 OPTIONS 预检请求
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, HEAD, OPTIONS';
            add_header 'Access-Control-Allow-Headers' '*';
            add_header 'Content-Length' 0;
            add_header 'Content-Type' 'text/plain';
            return 204;
        }

        # 反向代理到 MinIO
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

重启 Nginx:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 方案 3：Apache CORS 配置

如果使用 Apache，在 `.htaccess` 或 VirtualHost 配置中添加：

```apache
<IfModule mod_headers.c>
    Header set Access-Control-Allow-Origin "*"
    Header set Access-Control-Allow-Methods "GET, HEAD, OPTIONS"
    Header set Access-Control-Allow-Headers "*"
    Header set Access-Control-Expose-Headers "Content-Length, Content-Range, ETag"
    Header set Access-Control-Max-Age "3600"
</IfModule>
```

---

## 方案 4：简单 HTTP 服务器（Python）

如果只是简单的文件服务，可以用 Python 快速搭建：

```python
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        SimpleHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    httpd = HTTPServer(('0.0.0.0', 9000), CORSRequestHandler)
    print("Server running on port 9000...")
    httpd.serve_forever()
```

---

## 验证 CORS 配置

### 使用 curl 测试

```bash
curl -I -X GET http://118.24.188.137:9000/lxh/mesh.ply \
  -H "Origin: https://mihoutao-liu.github.io"
```

应该看到响应头中包含：
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
```

### 使用浏览器控制台测试

```javascript
fetch('http://118.24.188.137:9000/lxh/mesh.ply', {
  method: 'HEAD'
})
  .then(response => {
    console.log('✅ CORS 配置成功！');
    console.log('Headers:', [...response.headers.entries()]);
  })
  .catch(error => {
    console.error('❌ CORS 配置失败:', error);
  });
```

---

## 常见问题

### Q1: 配置后仍然报 CORS 错误？
**A:** 确保：
1. 清除浏览器缓存
2. 服务器已重启
3. 防火墙允许端口访问
4. Origin 设置为 `*` 或包含您的域名

### Q2: OPTIONS 预检请求失败？
**A:** 确保服务器正确处理 OPTIONS 请求，返回 200 或 204 状态码。

### Q3: 混合内容警告（http vs https）？
**A:** GitHub Pages 使用 HTTPS，而您的服务器是 HTTP。建议：
- 为服务器配置 SSL 证书（Let's Encrypt 免费）
- 或在客户端添加混合内容处理

---

## 推荐配置（按优先级）

1. **最佳方案**: MinIO CORS + Nginx SSL
   - 配置 MinIO CORS
   - 使用 Nginx 反向代理
   - 配置 SSL 证书（Let's Encrypt）

2. **简单方案**: 仅 MinIO CORS
   - 直接配置 MinIO bucket CORS
   - 适合测试和开发

3. **临时方案**: 客户端 CORS 代理
   - 仅用于开发测试
   - 生产环境必须配置服务器端

---

## 需要帮助？

如果遇到问题，请提供：
1. 服务器类型（MinIO/Nginx/Apache）
2. 错误信息截图
3. curl 测试结果

