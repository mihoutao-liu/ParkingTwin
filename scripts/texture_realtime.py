#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时纹理化 - 带轨迹验证的OpenMVS增强版
============================================================================
分两个阶段：
1. 轨迹验证阶段：显示轨迹，允许交互式调整（参考 verify_trajectory_from_txt.py）
2. 纹理化阶段：确认轨迹后开始实时纹理化

使用 OpenCV 标准相机坐标系（无额外变换）
"""
import sys
import os
from pathlib import Path

# Windows: 添加NVIDIA DLL目录（GPU加速所需）
if os.name == 'nt':
    user_site = Path(os.path.expanduser('~')) / 'AppData' / 'Roaming' / 'Python' / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'site-packages'
    nvidia_dir = user_site / 'nvidia'
    if nvidia_dir.exists():
        for pkg_dir in nvidia_dir.iterdir():
            if pkg_dir.is_dir():
                bin_dir = pkg_dir / 'bin'
                if bin_dir.exists() and hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(str(bin_dir))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import open3d as o3d
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import time
import threading
from collections import deque
from datetime import datetime
import yaml
import argparse

# GPU加速支持
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    GPU_DEVICE_COUNT = 0

print("="*70)
print("实时纹理化 - 带轨迹验证的OpenMVS增强版 (GPU加速)")
print("="*70)

# ============================================================================
# 命令行参数解析
# ============================================================================
parser = argparse.ArgumentParser(description='实时纹理化程序')
parser.add_argument('--config', type=str, default=None, 
                    help='配置文件路径 (YAML格式，例如：config_ground_clarity.yaml)')
args = parser.parse_args()

# ============================================================================
# 加载配置文件（如果指定）
# ============================================================================
def load_config(config_path):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✅ 已加载配置文件: {config_path}")
    return config

if args.config:
    cfg = load_config(args.config)
    print(f"📝 配置文件: {args.config}")
else:
    cfg = None
    print("📝 使用默认配置（代码中的参数）")

# ============================================================================
# 配置（从YAML加载或使用默认值）
# ============================================================================
mesh_path = cfg['paths']['mesh_path'] if cfg else "Datasets/eslam_data/tiles/mesh_tile_00.ply"
pose_dir = cfg['paths']['pose_dir'] if cfg else "Datasets/eslam_data/pose"
first_pose_txt = cfg['paths']['first_pose_txt'] if cfg else "Datasets/eslam_data/first_pose_osm.txt"
K_npz = cfg['paths']['K_npz'] if cfg else "Datasets/eslam_data/K_rectified.npz"
rgb_dir = cfg['paths']['rgb_dir'] if cfg else "Datasets/eslam_data/color"
depth_dir = cfg['paths']['depth_dir'] if cfg else "Datasets/eslam_data/depth"

# 可视化参数
SAMPLE_RATE = cfg['visualization']['sample_rate'] if cfg else 20
ARROW_SCALE = cfg['visualization']['arrow_scale'] if cfg else 1.0
ARROW_LENGTH = cfg['visualization']['arrow_length'] if cfg else 2.0
SHOW_CAMERA_FRUSTUM = cfg['visualization']['show_camera_frustum'] if cfg else True
VISUALIZATION_UPDATE_RATE = cfg['visualization']['update_rate'] if cfg else 10

# 纹理化参数
FRAME_SAMPLE_RATE = cfg['texturing']['frame_sample_rate'] if cfg else 1
MAX_IMAGES = cfg['texturing']['max_images'] if cfg else 10000
UPDATE_INTERVAL = cfg['texturing']['update_interval'] if cfg else 0.01

# OpenMVS增强参数
DEPTH_THRESHOLD = cfg['openmvs']['depth_threshold'] if cfg else 1.5
DEPTH_SCALE = cfg['openmvs']['depth_scale'] if cfg else 1000.0
MIN_DEPTH = cfg['openmvs']['min_depth'] if cfg else 0.1
MAX_DEPTH = cfg['openmvs']['max_depth'] if cfg else 20.0
ANGLE_THRESHOLD_DEG = cfg['openmvs']['angle_threshold_deg'] if cfg else 360
USE_DEPTH_CONSISTENCY = cfg['openmvs']['use_depth_consistency'] if cfg else True
USE_ANGLE_WEIGHTING = cfg['openmvs']['use_angle_weighting'] if cfg else False
USE_EXPOSURE_COMP = cfg['openmvs']['use_exposure_comp'] if cfg else True
DEBUG_MODE = cfg['openmvs']['debug_mode'] if cfg else True

# 自适应深度检测参数（新增）
USE_ADAPTIVE_DEPTH = cfg['openmvs'].get('use_adaptive_depth', False) if cfg else False
FLOOR_NORMAL_THRESHOLD = cfg['openmvs'].get('floor_normal_threshold', 0.7) if cfg else 0.7
FLOOR_DEPTH_FRONT = cfg['openmvs'].get('floor_depth_front', 0.3) if cfg else 0.3
FLOOR_DEPTH_BACK = cfg['openmvs'].get('floor_depth_back', 0.08) if cfg else 0.08
WALL_DEPTH_FRONT = cfg['openmvs'].get('wall_depth_front', 0.5) if cfg else 0.5
WALL_DEPTH_BACK = cfg['openmvs'].get('wall_depth_back', 0.15) if cfg else 0.15

# ============================================================================
# 车辆检测参数（四模态几何检测）
# ============================================================================
if cfg and 'vehicle_detection' in cfg:
    vd_cfg = cfg['vehicle_detection']
    USE_VEHICLE_DETECTION = vd_cfg.get('enable', False)
    # 1. 法向检测
    USE_GROUND_NORMAL = vd_cfg.get('use_ground_normal', True)
    GROUND_NORMAL_THRESHOLD = vd_cfg.get('ground_normal_threshold', 0.94)
    # 2. 高度过滤
    USE_HEIGHT_FILTER = vd_cfg.get('use_height_filter', True)
    VEHICLE_HEIGHT_MIN = vd_cfg.get('vehicle_height_min', 0.5)
    VEHICLE_HEIGHT_MAX = vd_cfg.get('vehicle_height_max', 2.5)
    # 3. 深度不连续
    USE_DEPTH_DISCONTINUITY = vd_cfg.get('use_depth_discontinuity', True)
    DEPTH_GRADIENT_THRESHOLD = vd_cfg.get('depth_gradient_threshold', 1.0)
    # 4. TSDF深度一致性（新增！）
    USE_TSDF_DEPTH_CONSISTENCY = vd_cfg.get('use_depth_consistency', False)
    DEPTH_DIFF_THRESHOLD = vd_cfg.get('depth_diff_threshold', 0.3)
    DEPTH_NOISE_TOLERANCE = vd_cfg.get('depth_noise_tolerance', 0.05)
    # 综合设置
    REQUIRE_ALL_CUES = vd_cfg.get('require_all_cues', True)
    VEHICLE_MASK_DILATION = vd_cfg.get('mask_dilation', 5)
    SAVE_VEHICLE_MASKS = vd_cfg.get('save_masks', False)
else:
    # 默认关闭
    USE_VEHICLE_DETECTION = False
    USE_GROUND_NORMAL = True
    GROUND_NORMAL_THRESHOLD = 0.94
    USE_HEIGHT_FILTER = True
    VEHICLE_HEIGHT_MIN = 0.5
    VEHICLE_HEIGHT_MAX = 2.5
    USE_DEPTH_DISCONTINUITY = True
    DEPTH_GRADIENT_THRESHOLD = 1.0
    USE_TSDF_DEPTH_CONSISTENCY = False
    DEPTH_DIFF_THRESHOLD = 0.3
    DEPTH_NOISE_TOLERANCE = 0.05
    REQUIRE_ALL_CUES = True
    VEHICLE_MASK_DILATION = 5
    SAVE_VEHICLE_MASKS = False

# 向后兼容旧参数
USE_VEHICLE_REMOVAL = USE_VEHICLE_DETECTION

# 固定高度参数
FIX_CAMERA_HEIGHT = cfg['camera']['fix_camera_height'] if cfg else True
FIXED_HEIGHT = cfg['camera']['fixed_height'] if cfg else None
# 强制相机水平参数（新增）
FORCE_CAMERA_HORIZONTAL = cfg['camera'].get('force_camera_horizontal', False) if cfg else False

# 后处理参数：空白区域填充
FILL_EMPTY_VERTICES = cfg.get('post_processing', {}).get('fill_empty_vertices', False) if cfg else False
FILL_METHOD = cfg.get('post_processing', {}).get('fill_method', 'knn') if cfg else 'knn'
KNN_NEIGHBORS = cfg.get('post_processing', {}).get('knn_neighbors', 8) if cfg else 8

# 模型保存参数
SAVE_TEXTURED_MESH = cfg.get('output', {}).get('save_textured_mesh', True) if cfg else True
OUTPUT_DIR = cfg.get('output', {}).get('output_dir', 'output/textured_meshes') if cfg else 'output/textured_meshes'
OUTPUT_FILENAME = cfg.get('output', {}).get('output_filename', 'textured_mesh.ply') if cfg else 'textured_mesh.ply'
AUTO_TIMESTAMP = cfg.get('output', {}).get('auto_timestamp', True) if cfg else True

# GPU加速参数
USE_GPU = cfg['gpu']['use_gpu'] if cfg else True
GPU_DEVICE_ID = cfg['gpu']['device_id'] if cfg else 0

# 旋转配置保存/加载
ROTATION_CONFIG_FILE = cfg['rotation']['config_file'] if cfg else "Datasets/eslam_data/rotation_config.json"
AUTO_LOAD_ROTATION = cfg['rotation']['auto_load'] if cfg else True
DEFAULT_ROTATION = tuple(cfg['rotation']['default_rotation']) if cfg else (180, 0, 0)

# ============================================================================
# 纹理质量增强参数 - 第1步：图像质量评估与过滤
# ============================================================================
USE_IMAGE_QUALITY_FILTER = cfg['step1_quality_filter']['enable'] if cfg else True
IMAGE_QUALITY_THRESHOLD = cfg['step1_quality_filter']['quality_threshold'] if cfg else 30.0
SHARPNESS_THRESHOLD = cfg['step1_quality_filter']['sharpness_threshold'] if cfg else 30.0
MAX_OVEREXPOSURE = cfg['step1_quality_filter']['max_overexposure'] if cfg else 0.15
MAX_UNDEREXPOSURE = cfg['step1_quality_filter']['max_underexposure'] if cfg else 0.15
SHOW_QUALITY_STATS = cfg['step1_quality_filter']['show_quality_stats'] if cfg else True

# ============================================================================
# 纹理质量增强参数 - 第2步：图像预处理增强
# ============================================================================
USE_IMAGE_ENHANCEMENT = cfg['step2_image_enhancement']['enable'] if cfg else True
USE_UNSHARP_MASK = cfg['step2_image_enhancement']['use_unsharp_mask'] if cfg else True
UNSHARP_RADIUS = cfg['step2_image_enhancement']['unsharp_radius'] if cfg else 2.0
UNSHARP_AMOUNT = cfg['step2_image_enhancement']['unsharp_amount'] if cfg else 1.5
USE_BILATERAL_FILTER = cfg['step2_image_enhancement']['use_bilateral_filter'] if cfg else True
BILATERAL_D = cfg['step2_image_enhancement']['bilateral_d'] if cfg else 5
BILATERAL_SIGMA_COLOR = cfg['step2_image_enhancement']['bilateral_sigma_color'] if cfg else 75
BILATERAL_SIGMA_SPACE = cfg['step2_image_enhancement']['bilateral_sigma_space'] if cfg else 75
USE_CLAHE = cfg['step2_image_enhancement']['use_clahe'] if cfg else True
CLAHE_CLIP_LIMIT = cfg['step2_image_enhancement']['clahe_clip_limit'] if cfg else 2.0
CLAHE_TILE_SIZE = cfg['step2_image_enhancement']['clahe_tile_size'] if cfg else 8

# 第3步优化参数：双三次插值
USE_BICUBIC_INTERPOLATION = cfg['step3_bicubic']['enable'] if cfg else True
BICUBIC_A = cfg['step3_bicubic']['bicubic_a'] if cfg else -0.5

# 第4步优化参数：智能视角选择与加权
USE_SMART_VIEW_WEIGHTING = cfg['step4_view_weighting']['enable'] if cfg else True
VIEW_ANGLE_WEIGHT = cfg['step4_view_weighting']['view_angle_weight'] if cfg else 0.4
DISTANCE_WEIGHT = cfg['step4_view_weighting']['distance_weight'] if cfg else 0.3
IMAGE_QUALITY_WEIGHT = cfg['step4_view_weighting']['image_quality_weight'] if cfg else 0.3
MAX_VIEW_ANGLE_DEG = cfg['step4_view_weighting']['max_view_angle_deg'] if cfg else 75.0
DISTANCE_FALLOFF = cfg['step4_view_weighting']['distance_falloff'] if cfg else 2.0
MIN_EFFECTIVE_WEIGHT = cfg['step4_view_weighting']['min_effective_weight'] if cfg else 0.1

# ---------- 第5步优化：接缝平滑配置 ----------
USE_SEAM_SMOOTHING = cfg['step5_seam_smoothing']['enable'] if cfg else True
VARIANCE_THRESHOLD = cfg['step5_seam_smoothing']['variance_threshold'] if cfg else 0.01
SMOOTHING_STRENGTH = cfg['step5_seam_smoothing']['smoothing_strength'] if cfg else 0.5
SEAM_K_NEIGHBORS = cfg['step5_seam_smoothing']['k_neighbors'] if cfg else 15

# ---------- 第6步优化：LAB色彩空间配置 ----------
USE_LAB_COLOR_SPACE = cfg['step6_lab_color']['enable'] if cfg else True
LAB_L_WEIGHT = cfg['step6_lab_color']['l_weight'] if cfg else 0.5
LAB_NORMALIZE_L = cfg['step6_lab_color']['normalize_l'] if cfg else True
LAB_L_CLIP_PERCENTILE = cfg['step6_lab_color']['l_clip_percentile'] if cfg else 2.0

# ---------- 第7步优化：亚像素精度投影配置 ----------
USE_SUBPIXEL_PRECISION = cfg['step7_subpixel']['enable'] if cfg else True
USE_FLOAT64_PROJECTION = cfg['step7_subpixel']['use_float64'] if cfg else True
SUBPIXEL_WEIGHT_MODE = cfg['step7_subpixel']['weight_mode'] if cfg else "bilinear"
PRESERVE_SUBPIXEL_WEIGHT = cfg['step7_subpixel']['preserve_weight'] if cfg else True
PROJECTION_EPSILON = cfg['step7_subpixel']['projection_epsilon'] if cfg else 1e-10

# ---------- 第8步优化：纹理后处理配置 ----------
USE_POST_PROCESSING = cfg['step8_post_processing']['enable'] if cfg else True
# 异常值检测
USE_OUTLIER_DETECTION = cfg['step8_post_processing']['outlier_detection']['enable'] if cfg else True
OUTLIER_DETECTION_METHOD = cfg['step8_post_processing']['outlier_detection']['method'] if cfg else "both"
OUTLIER_ZSCORE_THRESHOLD = cfg['step8_post_processing']['outlier_detection']['zscore_threshold'] if cfg else 3.0
OUTLIER_LOCAL_WINDOW = cfg['step8_post_processing']['outlier_detection']['local_window'] if cfg else 5
OUTLIER_LOCAL_THRESHOLD = cfg['step8_post_processing']['outlier_detection']['local_threshold'] if cfg else 2.5
# 边缘保持平滑
USE_EDGE_PRESERVING_SMOOTH = cfg['step8_post_processing']['edge_preserving']['enable'] if cfg else True
SMOOTH_METHOD = cfg['step8_post_processing']['edge_preserving']['method'] if cfg else "bilateral"
BILATERAL_SIGMA_SPATIAL = cfg['step8_post_processing']['edge_preserving']['bilateral_sigma_spatial'] if cfg else 5.0
BILATERAL_SIGMA_COLOR = cfg['step8_post_processing']['edge_preserving']['bilateral_sigma_color'] if cfg else 25.0
ANISOTROPIC_ITERATIONS = cfg['step8_post_processing']['edge_preserving']['anisotropic_iterations'] if cfg else 10
ANISOTROPIC_KAPPA = cfg['step8_post_processing']['edge_preserving']['anisotropic_kappa'] if cfg else 50.0
# 色彩一致性校正
USE_COLOR_CORRECTION = cfg['step8_post_processing']['color_correction']['enable'] if cfg else True
COLOR_CORRECTION_METHOD = cfg['step8_post_processing']['color_correction']['method'] if cfg else "histogram"
HISTOGRAM_MATCH_PERCENTILE = cfg['step8_post_processing']['color_correction']['histogram_match_percentile'] if cfg else 5.0
COLOR_TRANSFER_PRESERVE_LUMINANCE = cfg['step8_post_processing']['color_correction']['transfer_preserve_luminance'] if cfg else True

# 检测GPU可用性并设置
if USE_GPU and GPU_AVAILABLE:
    try:
        device = cp.cuda.Device(GPU_DEVICE_ID)
        device.use()
        USE_GPU = True
        print(f"\n🚀 GPU加速已启用")
        # 获取GPU名称（兼容不同CuPy版本）
        try:
            gpu_name = device.attributes.get('Name', b'Unknown GPU')
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            print(f"  - GPU设备: {gpu_name}")
        except:
            print(f"  - GPU设备: Device {GPU_DEVICE_ID}")
        # 获取显存信息
        try:
            mem_total = device.mem_info[1] / 10243
            print(f"  - 显存: {mem_total:.1f} GB")
        except:
            print(f"  - 显存: 可用")
    except Exception as e:
        print(f"\n⚠️  GPU初始化失败，降级到CPU模式: {e}")
        USE_GPU = False
elif USE_GPU and not GPU_AVAILABLE:
    print(f"\n⚠️  CuPy未安装或GPU不可用，使用CPU模式")
    print(f"     安装命令: pip install cupy-cuda12x")
    USE_GPU = False
else:
    print(f"\n💻 使用CPU模式")

print(f"\n配置:")
print(f"  - 计算设备: {'GPU' if USE_GPU else 'CPU'}")
print(f"  - 采样率: 每{FRAME_SAMPLE_RATE}帧")
print(f"  - 最大帧数: {MAX_IMAGES}")
print(f"  - 固定摄像头高度: {'开启' if FIX_CAMERA_HEIGHT else '关闭'}")
if FIX_CAMERA_HEIGHT and FIXED_HEIGHT is not None:
    print(f"    └─ 固定高度值: {FIXED_HEIGHT:.3f}m")
print(f"  - 强制相机水平: {'开启' if FORCE_CAMERA_HORIZONTAL else '关闭'}")
print(f"  - 深度一致性: {'开启' if USE_DEPTH_CONSISTENCY else '关闭'}")
print(f"  - 视角加权: {'开启' if USE_ANGLE_WEIGHTING else '关闭'}")
print(f"  - 曝光补偿: {'开启' if USE_EXPOSURE_COMP else '关闭'}")
print(f"  - 车辆检测（四模态）: {'✓ 开启' if USE_VEHICLE_DETECTION else '关闭'}")
if USE_VEHICLE_DETECTION:
    enabled_cues = []
    if USE_GROUND_NORMAL:
        enabled_cues.append(f"法向(>{GROUND_NORMAL_THRESHOLD:.2f})")
    if USE_HEIGHT_FILTER:
        enabled_cues.append(f"高度({VEHICLE_HEIGHT_MIN}-{VEHICLE_HEIGHT_MAX}m)")
    if USE_DEPTH_DISCONTINUITY:
        enabled_cues.append(f"梯度(>{DEPTH_GRADIENT_THRESHOLD}m/px)")
    if USE_TSDF_DEPTH_CONSISTENCY:
        enabled_cues.append(f"TSDF深度(>{DEPTH_DIFF_THRESHOLD}m)")
    print(f"    └─ 启用线索: {' + '.join(enabled_cues)}")
    print(f"    └─ 融合方式: {'AND (所有满足)' if REQUIRE_ALL_CUES else 'OR (任意满足)'}")
print(f"  - 图像质量过滤: {'✓ 开启 (第1步优化)' if USE_IMAGE_QUALITY_FILTER else '关闭'}")
if USE_IMAGE_QUALITY_FILTER:
    print(f"    └─ 质量阈值: {IMAGE_QUALITY_THRESHOLD:.0f}")
    print(f"    └─ 清晰度阈值: {SHARPNESS_THRESHOLD:.0f}")
    print(f"    └─ 最大过曝: {MAX_OVEREXPOSURE*100:.0f}%")
    print(f"    └─ 最大欠曝: {MAX_UNDEREXPOSURE*100:.0f}%")
print(f"  - 图像预处理增强: {'✓ 开启 (第2步优化)' if USE_IMAGE_ENHANCEMENT else '关闭'}")
if USE_IMAGE_ENHANCEMENT:
    enhancement_methods = []
    if USE_BILATERAL_FILTER:
        enhancement_methods.append(f"双边滤波(d={BILATERAL_D})")
    if USE_CLAHE:
        enhancement_methods.append(f"CLAHE(clip={CLAHE_CLIP_LIMIT})")
    if USE_UNSHARP_MASK:
        enhancement_methods.append(f"非锐化(r={UNSHARP_RADIUS}, a={UNSHARP_AMOUNT})")
    if enhancement_methods:
        print(f"    └─ {' + '.join(enhancement_methods)}")
print(f"  - 双三次插值: {'✓ 开启 (第3步优化, a={BICUBIC_A})' if USE_BICUBIC_INTERPOLATION else '关闭 (使用双线性)'}")

# 第4步优化配置显示
if USE_SMART_VIEW_WEIGHTING:
    print(f"  - 智能视角加权: ✓ 开启 (第4步优化)")
    print(f"    └─ 权重配置: 视角{VIEW_ANGLE_WEIGHT:.1f} + 距离{DISTANCE_WEIGHT:.1f} + 质量{IMAGE_QUALITY_WEIGHT:.1f}")
    print(f"    └─ 最大视角: {MAX_VIEW_ANGLE_DEG:.0f}°, 距离衰减: {DISTANCE_FALLOFF:.1f}, 最小权重: {MIN_EFFECTIVE_WEIGHT:.2f}")
elif USE_ANGLE_WEIGHTING:
    print(f"  - 传统视角加权: 开启 (阈值: {ANGLE_THRESHOLD_DEG}°)")
else:
    print(f"  - 视角加权: 关闭")

# 第5步优化配置显示
if USE_SEAM_SMOOTHING:
    print(f"  - 接缝平滑: ✓ 开启 (第5步优化)")
    print(f"    └─ 方差阈值: {VARIANCE_THRESHOLD:.3f}, 平滑强度: {SMOOTHING_STRENGTH:.1f}, 邻域: {SEAM_K_NEIGHBORS}顶点")
else:
    print(f"  - 接缝平滑: 关闭")

# 第6步优化配置显示
if USE_LAB_COLOR_SPACE:
    print(f"  - LAB色彩空间: ✓ 开启 (第6步优化)")
    print(f"    └─ L通道权重: {LAB_L_WEIGHT}, 归一化: {'是' if LAB_NORMALIZE_L else '否'}, 裁剪百分位: {LAB_L_CLIP_PERCENTILE:.1f}%")
else:
    print(f"  - LAB色彩空间: 关闭 (RGB空间)")

# 第7步优化配置显示
if USE_SUBPIXEL_PRECISION:
    print(f"  - 亚像素精度投影: ✓ 开启 (第7步优化)")
    print(f"    └─ 投影精度: {'float64' if USE_FLOAT64_PROJECTION else 'float32'}, 权重模式: {SUBPIXEL_WEIGHT_MODE}")
    print(f"    └─ 保留亚像素权重: {'是' if PRESERVE_SUBPIXEL_WEIGHT else '否'}, 数值阈值: {PROJECTION_EPSILON:.0e}")
else:
    print(f"  - 亚像素精度投影: 关闭 (标准int投影)")

# ============================================================================
# 图像质量评估函数
# ============================================================================
def assess_image_quality(rgb_img):
    """
    评估图像质量：清晰度和曝光
    
    参数:
        rgb_img: RGB图像，取值范围0-1，形状(H, W, 3)
    
    返回:
        quality_score: 综合质量分数
        sharpness: 清晰度（拉普拉斯方差）
        overexposed: 过曝像素比例
        underexposed: 欠曝像素比例
        contrast: 对比度（标准差）
    """
    # 转换为灰度图用于分析
    gray = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 1. 拉普拉斯方差 - 检测模糊（清晰度指标）
    # 模糊图像的拉普拉斯方差较低
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # 2. 直方图分析 - 检测曝光问题
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = hist.sum()
    
    # 过曝：灰度值>240的像素比例
    overexposed = np.sum(hist[240:]) / total_pixels
    
    # 欠曝：灰度值<15的像素比例
    underexposed = np.sum(hist[:15]) / total_pixels
    
    # 3. 对比度（标准差）
    contrast = gray.std()
    
    # 4. 计算综合质量分数
    # 公式：清晰度 × (1 - 曝光问题) × (对比度因子)
    exposure_penalty = 1.0 - overexposed - underexposed
    contrast_factor = min(contrast / 50.0, 1.0)  # 标准化到0-1
    
    quality_score = sharpness * exposure_penalty * contrast_factor
    
    return quality_score, sharpness, overexposed, underexposed, contrast

# ============================================================================
# 图像增强函数（第2步优化）
# ============================================================================
def apply_unsharp_mask(image, radius=2.0, amount=1.5):
    """
    非锐化掩蔽（Unsharp Masking）- 增强图像细节和边缘
    
    原理：原始图像 + (原始图像 - 模糊图像) × 强度
    
    参数:
        image: RGB图像，取值0-1，形状(H, W, 3)
        radius: 高斯模糊半径（越大增强越强，建议1.0-3.0）
        amount: 锐化强度（越大越锐利，建议1.0-2.0）
    
    返回:
        增强后的图像，取值0-1
    """
    # 转换到uint8进行处理
    img_uint8 = (image * 255).astype(np.uint8)
    
    # 创建高斯模糊版本
    kernel_size = int(2 * np.ceil(2 * radius) + 1)  # 保证奇数
    blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), radius)
    
    # 计算锐化掩模：原图 - 模糊图
    mask = cv2.subtract(img_uint8, blurred)
    
    # 应用锐化：原图 + 掩模 × 强度
    sharpened = cv2.addWeighted(img_uint8, 1.0, mask, amount, 0)
    
    # 转换回0-1范围
    return sharpened.astype(np.float32) / 255.0


def apply_bilateral_filter(image, d=5, sigma_color=75, sigma_space=75):
    """
    双边滤波 - 去噪同时保持边缘
    
    原理：同时考虑空间距离和颜色相似度的加权平均
    
    参数:
        image: RGB图像，取值0-1，形状(H, W, 3)
        d: 滤波直径（建议5-9）
        sigma_color: 颜色空间标准差（建议50-100，越大颜色差异越被忽略）
        sigma_space: 坐标空间标准差（建议50-100，越大像素影响范围越广）
    
    返回:
        去噪后的图像，取值0-1
    """
    # 转换到uint8
    img_uint8 = (image * 255).astype(np.uint8)
    
    # 应用双边滤波
    filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
    
    # 转换回0-1范围
    return filtered.astype(np.float32) / 255.0


def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    CLAHE（对比度受限自适应直方图均衡化）- 增强局部对比度
    
    原理：在小区域内进行直方图均衡化，同时限制对比度增强
    
    参数:
        image: RGB图像，取值0-1，形状(H, W, 3)
        clip_limit: 对比度限制（1.0-4.0，越大对比度越强）
        tile_size: 网格大小（8-16，越小局部对比度越强）
    
    返回:
        增强后的图像，取值0-1
    """
    # 转换到uint8
    img_uint8 = (image * 255).astype(np.uint8)
    
    # 转换到LAB色彩空间（只对亮度通道增强）
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 创建CLAHE对象并应用到L通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)
    
    # 合并通道
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # 转换回RGB
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # 转换回0-1范围
    return rgb_enhanced.astype(np.float32) / 255.0


def enhance_image(image):
    """
    综合图像增强函数 - 整合多种增强技术
    
    处理流程：
    1. 双边滤波去噪（可选）
    2. CLAHE对比度增强（可选）
    3. 非锐化掩蔽锐化（可选）
    
    参数:
        image: RGB图像，取值0-1，形状(H, W, 3)
    
    返回:
        增强后的图像，取值0-1
    """
    if not USE_IMAGE_ENHANCEMENT:
        return image
    
    enhanced = image.copy()
    
    # 第1步：去噪（双边滤波） - 先去噪再增强效果更好
    if USE_BILATERAL_FILTER:
        enhanced = apply_bilateral_filter(
            enhanced, 
            d=BILATERAL_D,
            sigma_color=BILATERAL_SIGMA_COLOR,
            sigma_space=BILATERAL_SIGMA_SPACE
        )
    
    # 第2步：对比度增强（CLAHE）
    if USE_CLAHE:
        enhanced = apply_clahe(
            enhanced,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_size=CLAHE_TILE_SIZE
        )
    
    # 第3步：锐化（非锐化掩蔽）
    if USE_UNSHARP_MASK:
        enhanced = apply_unsharp_mask(
            enhanced,
            radius=UNSHARP_RADIUS,
            amount=UNSHARP_AMOUNT
        )
    
    # 确保值域在0-1
    enhanced = np.clip(enhanced, 0.0, 1.0)
    
    return enhanced

# ============================================================================
# 第7步优化：亚像素精度投影函数
# ============================================================================
def project_vertices_subpixel(vertices_cam, K, H, W, use_float64=True):
    """
    高精度投影3D点到2D像素坐标（亚像素级精度）
    
    参数:
        vertices_cam: 相机坐标系下的3D点 (N, 3)，float32/float64
        K: 相机内参矩阵 (3, 3)
        H, W: 图像高度和宽度
        use_float64: 是否使用float64提高精度
    
    返回:
        u_f, v_f: 浮点像素坐标 (N,)，保留完整亚像素精度
    """
    if use_float64:
        # 转换为float64提高精度
        vertices_cam = vertices_cam.astype(np.float64)
        K = K.astype(np.float64)
    
    # 透视投影（保持高精度）
    points_2d = K @ vertices_cam.T  # (3, N)
    
    # 防止除零（使用更小的epsilon提高精度）
    depths = points_2d[2, :]
    depths = np.where(np.abs(depths) < PROJECTION_EPSILON, PROJECTION_EPSILON, depths)
    
    # 归一化到像素坐标（完整浮点精度）
    u_f = points_2d[0, :] / depths
    v_f = points_2d[1, :] / depths
    
    # Y轴翻转（OpenCV坐标系）
    v_f = H - 1.0 - v_f
    
    return u_f, v_f


def project_vertices_subpixel_gpu(vertices_cam_gpu, K_gpu, H, W, use_float64=True):
    """
    高精度投影3D点到2D像素坐标（GPU版本，亚像素级精度）
    
    参数:
        vertices_cam_gpu: 相机坐标系下的3D点 (N, 3)，CuPy数组
        K_gpu: 相机内参矩阵 (3, 3)，CuPy数组
        H, W: 图像高度和宽度
        use_float64: 是否使用float64提高精度
    
    返回:
        u_f_gpu, v_f_gpu: 浮点像素坐标 (N,)，CuPy数组，保留完整亚像素精度
    """
    xp = cp
    if use_float64:
        # 转换为float64提高精度
        vertices_cam_gpu = vertices_cam_gpu.astype(xp.float64)
        K_gpu = K_gpu.astype(xp.float64)
    
    # 透视投影（保持高精度）
    points_2d_gpu = K_gpu @ vertices_cam_gpu.T  # (3, N)
    
    # 防止除零
    depths = points_2d_gpu[2, :]
    epsilon = xp.float64(PROJECTION_EPSILON) if use_float64 else xp.float32(PROJECTION_EPSILON)
    depths = xp.where(xp.abs(depths) < epsilon, epsilon, depths)
    
    # 归一化到像素坐标（完整浮点精度）
    u_f_gpu = points_2d_gpu[0, :] / depths
    v_f_gpu = points_2d_gpu[1, :] / depths
    
    # Y轴翻转（OpenCV坐标系）
    v_f_gpu = H - 1.0 - v_f_gpu
    
    # 转回float32（如果需要）
    if use_float64:
        u_f_gpu = u_f_gpu.astype(xp.float32)
        v_f_gpu = v_f_gpu.astype(xp.float32)
    
    return u_f_gpu, v_f_gpu


def compute_subpixel_weights(u_f, v_f, mode="bilinear"):
    """
    计算亚像素位置的插值权重（不进行量化）
    
    参数:
        u_f, v_f: 浮点像素坐标 (N,)
        mode: 插值模式 "bilinear" 或 "bicubic"
    
    返回:
        weights: 亚像素权重信息字典
            - "u_int", "v_int": 整数坐标
            - "u_frac", "v_frac": 小数部分（亚像素偏移）
            - "mode": 插值模式
    """
    # 整数部分和小数部分（保留完整精度）
    u_int = np.floor(u_f).astype(np.int32)
    v_int = np.floor(v_f).astype(np.int32)
    u_frac = u_f - u_int  # 亚像素偏移 [0, 1)
    v_frac = v_f - v_int  # 亚像素偏移 [0, 1)
    
    weights = {
        "u_int": u_int,
        "v_int": v_int,
        "u_frac": u_frac,  # 关键：保留小数精度
        "v_frac": v_frac,
        "mode": mode
    }
    
    return weights


def sample_with_subpixel_weights(img, weights, H, W):
    """
    使用亚像素权重进行图像采样
    
    参数:
        img: 输入图像 (H, W, C)
        weights: compute_subpixel_weights返回的权重字典
        H, W: 图像尺寸
    
    返回:
        colors: 采样颜色 (N, C)
    """
    u_int = weights["u_int"]
    v_int = weights["v_int"]
    u_frac = weights["u_frac"]
    v_frac = weights["v_frac"]
    mode = weights["mode"]
    
    if mode == "bilinear":
        # 双线性插值（4个邻域像素）
        u0, v0 = u_int, v_int
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)
        u0 = np.maximum(u0, 0)
        v0 = np.maximum(v0, 0)
        
        # 4个角点
        c00 = img[v0, u0]
        c10 = img[v0, u1]
        c01 = img[v1, u0]
        c11 = img[v1, u1]
        
        # 双线性插值（保留亚像素权重精度）
        wu = u_frac[:, np.newaxis]  # (N, 1)
        wv = v_frac[:, np.newaxis]  # (N, 1)
        
        colors = (c00 * (1 - wu) * (1 - wv) +
                  c10 * wu * (1 - wv) +
                  c01 * (1 - wu) * wv +
                  c11 * wu * wv)
        
    elif mode == "bicubic":
        # 双三次插值（16个邻域像素）- 复用现有函数
        colors = bicubic_interpolate(img, u_int + u_frac, v_int + v_frac, a=-0.5)
    
    else:
        raise ValueError(f"不支持的插值模式: {mode}")
    
    return colors


# ============================================================================
# 第8步优化：纹理后处理函数
# ============================================================================

def detect_outliers_statistical(colors, weights, threshold=3.0):
    """
    统计学方法检测颜色异常值（基于Z-score）
    
    参数:
        colors: 顶点颜色 (N, 3) float32/float64，范围0-1
        weights: 顶点权重 (N,)，用于加权统计
        threshold: Z-score阈值（标准差倍数），默认3.0
    
    返回:
        outlier_mask: 异常值掩码 (N,) bool，True表示异常值
    """
    if len(colors) == 0:
        return np.zeros(len(colors), dtype=bool)
    
    # 加权计算均值和标准差（每个通道独立）
    valid_mask = weights > 0
    if not valid_mask.any():
        return np.zeros(len(colors), dtype=bool)
    
    weights_norm = weights[valid_mask] / weights[valid_mask].sum()
    
    # 逐通道计算加权统计量
    outlier_mask = np.zeros(len(colors), dtype=bool)
    
    for c in range(colors.shape[1]):
        channel = colors[valid_mask, c]
        mean_c = np.average(channel, weights=weights_norm)
        var_c = np.average((channel - mean_c)**2, weights=weights_norm)
        std_c = np.sqrt(var_c) + 1e-8  # 防止除零
        
        # Z-score检测
        z_scores = np.abs((colors[:, c] - mean_c) / std_c)
        outlier_mask |= (z_scores > threshold)
    
    return outlier_mask


def detect_outliers_local(vertex_colors, vertex_positions, weights, 
                          k_neighbors=8, threshold=2.5):
    """
    局部邻域方法检测颜色异常值
    
    参数:
        vertex_colors: 顶点颜色 (N, 3)
        vertex_positions: 顶点3D坐标 (N, 3)
        weights: 顶点权重 (N,)
        k_neighbors: K近邻数量
        threshold: 局部异常阈值（标准差倍数）
    
    返回:
        outlier_mask: 异常值掩码 (N,) bool
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(vertex_colors) == 0 or len(vertex_colors) < k_neighbors:
        return np.zeros(len(vertex_colors), dtype=bool)
    
    valid_mask = weights > 0
    if valid_mask.sum() < k_neighbors:
        return np.zeros(len(vertex_colors), dtype=bool)
    
    # 构建KNN索引（只考虑有权重的顶点）
    valid_positions = vertex_positions[valid_mask]
    valid_colors = vertex_colors[valid_mask]
    
    # 避免k_neighbors超过有效顶点数
    k = min(k_neighbors, len(valid_positions) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(valid_positions)
    distances, indices = nbrs.kneighbors(valid_positions)
    
    # 计算每个顶点与其邻域的颜色偏差
    outlier_mask_local = np.zeros(valid_mask.sum(), dtype=bool)
    
    for i in range(len(valid_colors)):
        # 邻域索引（排除自己）
        neighbor_idx = indices[i, 1:]
        neighbor_colors = valid_colors[neighbor_idx]
        
        # 计算邻域颜色均值和标准差
        mean_color = neighbor_colors.mean(axis=0)
        std_color = neighbor_colors.std(axis=0) + 1e-8
        
        # 计算当前顶点与邻域的差异
        color_diff = np.abs(valid_colors[i] - mean_color) / std_color
        
        # 任一通道超过阈值则标记为异常
        if (color_diff > threshold).any():
            outlier_mask_local[i] = True
    
    # 将局部掩码映射回全局
    outlier_mask = np.zeros(len(vertex_colors), dtype=bool)
    outlier_mask[valid_mask] = outlier_mask_local
    
    return outlier_mask


def detect_outliers(vertex_colors, vertex_positions, weights, 
                    method="both", zscore_threshold=3.0, 
                    local_k=8, local_threshold=2.5):
    """
    综合异常值检测（统计学 + 局部邻域）
    
    参数:
        vertex_colors: 顶点颜色 (N, 3)
        vertex_positions: 顶点3D坐标 (N, 3)
        weights: 顶点权重 (N,)
        method: 检测方法 "statistical"/"local"/"both"
        zscore_threshold: 统计学Z-score阈值
        local_k: 局部邻域K近邻数量
        local_threshold: 局部异常阈值
    
    返回:
        outlier_mask: 异常值掩码 (N,) bool
        outlier_count: 检测到的异常值数量
    """
    if method == "statistical":
        outlier_mask = detect_outliers_statistical(vertex_colors, weights, zscore_threshold)
    elif method == "local":
        outlier_mask = detect_outliers_local(vertex_colors, vertex_positions, weights, 
                                             local_k, local_threshold)
    elif method == "both":
        # 两种方法的并集
        mask1 = detect_outliers_statistical(vertex_colors, weights, zscore_threshold)
        mask2 = detect_outliers_local(vertex_colors, vertex_positions, weights, 
                                      local_k, local_threshold)
        outlier_mask = mask1 | mask2
    else:
        raise ValueError(f"不支持的检测方法: {method}")
    
    outlier_count = outlier_mask.sum()
    return outlier_mask, outlier_count


def bilateral_filter_texture(vertex_colors, vertex_positions, weights,
                             sigma_spatial=5.0, sigma_color=0.1, k_neighbors=16):
    """
    双边滤波进行边缘保持平滑
    
    参数:
        vertex_colors: 顶点颜色 (N, 3) 范围0-1
        vertex_positions: 顶点3D坐标 (N, 3)
        weights: 顶点权重 (N,)
        sigma_spatial: 空间标准差（米）
        sigma_color: 颜色标准差（0-1范围）
        k_neighbors: K近邻数量
    
    返回:
        filtered_colors: 滤波后的颜色 (N, 3)
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(vertex_colors) == 0:
        return vertex_colors.copy()
    
    valid_mask = weights > 0
    if valid_mask.sum() < k_neighbors:
        return vertex_colors.copy()
    
    filtered_colors = vertex_colors.copy()
    
    # 构建KNN索引
    valid_positions = vertex_positions[valid_mask]
    valid_colors = vertex_colors[valid_mask]
    
    k = min(k_neighbors, len(valid_positions) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(valid_positions)
    distances, indices = nbrs.kneighbors(valid_positions)
    
    # 对每个有效顶点进行双边滤波
    filtered_valid = np.zeros_like(valid_colors)
    
    for i in range(len(valid_colors)):
        neighbor_idx = indices[i, 1:]
        neighbor_pos = valid_positions[neighbor_idx]
        neighbor_colors = valid_colors[neighbor_idx]
        spatial_dist = distances[i, 1:]
        
        # 计算空间权重（高斯）
        w_spatial = np.exp(-(spatial_dist2) / (2 * sigma_spatial2))
        
        # 计算颜色权重（高斯）
        color_dist = np.linalg.norm(neighbor_colors - valid_colors[i], axis=1)
        w_color = np.exp(-(color_dist2) / (2 * sigma_color2))
        
        # 综合权重
        w_total = w_spatial * w_color
        w_total = w_total / (w_total.sum() + 1e-8)
        
        # 加权平均
        filtered_valid[i] = (neighbor_colors * w_total[:, np.newaxis]).sum(axis=0)
    
    filtered_colors[valid_mask] = filtered_valid
    return filtered_colors


def anisotropic_diffusion_texture(vertex_colors, vertex_positions, weights,
                                  iterations=10, kappa=50.0, gamma=0.1):
    """
    各向异性扩散进行边缘保持平滑
    
    参数:
        vertex_colors: 顶点颜色 (N, 3)
        vertex_positions: 顶点3D坐标 (N, 3)
        weights: 顶点权重 (N,)
        iterations: 迭代次数
        kappa: 边缘敏感度参数（颜色梯度阈值，0-1）
        gamma: 扩散步长（0-0.25，越大收敛越快但可能不稳定）
    
    返回:
        diffused_colors: 扩散后的颜色 (N, 3)
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(vertex_colors) == 0:
        return vertex_colors.copy()
    
    valid_mask = weights > 0
    if valid_mask.sum() < 4:
        return vertex_colors.copy()
    
    # 构建邻接关系
    valid_positions = vertex_positions[valid_mask]
    valid_colors = vertex_colors[valid_mask].astype(np.float64)
    
    k = min(6, len(valid_positions) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(valid_positions)
    distances, indices = nbrs.kneighbors(valid_positions)
    
    # 迭代扩散
    diffused = valid_colors.copy()
    
    for iter_idx in range(iterations):
        new_diffused = diffused.copy()
        
        for i in range(len(diffused)):
            neighbor_idx = indices[i, 1:]
            neighbor_colors = diffused[neighbor_idx]
            
            # 计算颜色梯度
            gradients = neighbor_colors - diffused[i]
            gradient_mag = np.linalg.norm(gradients, axis=1)
            
            # Perona-Malik扩散系数（边缘抑制函数）
            c = np.exp(-(gradient_mag / kappa)**2)
            
            # 加权扩散
            flux = (gradients * c[:, np.newaxis]).sum(axis=0)
            new_diffused[i] = diffused[i] + gamma * flux
        
        diffused = new_diffused
    
    # 裁剪到有效范围
    diffused = np.clip(diffused, 0, 1)
    
    result = vertex_colors.copy()
    result[valid_mask] = diffused.astype(vertex_colors.dtype)
    return result


def edge_preserving_smooth(vertex_colors, vertex_positions, weights,
                           method="bilateral", **kwargs):
    """
    边缘保持平滑统一接口
    
    参数:
        vertex_colors: 顶点颜色 (N, 3)
        vertex_positions: 顶点3D坐标 (N, 3)
        weights: 顶点权重 (N,)
        method: 平滑方法 "bilateral"/"anisotropic"
        **kwargs: 方法特定参数
    
    返回:
        smoothed_colors: 平滑后的颜色 (N, 3)
    """
    if method == "bilateral":
        sigma_spatial = kwargs.get('sigma_spatial', 5.0)
        sigma_color = kwargs.get('sigma_color', 0.1)
        k_neighbors = kwargs.get('k_neighbors', 16)
        return bilateral_filter_texture(vertex_colors, vertex_positions, weights,
                                       sigma_spatial, sigma_color, k_neighbors)
    elif method == "anisotropic":
        iterations = kwargs.get('iterations', 10)
        kappa = kwargs.get('kappa', 50.0)
        gamma = kwargs.get('gamma', 0.1)
        return anisotropic_diffusion_texture(vertex_colors, vertex_positions, weights,
                                            iterations, kappa, gamma)
    else:
        raise ValueError(f"不支持的平滑方法: {method}")


def histogram_match_colors(source_colors, reference_colors, weights=None, 
                           clip_percentile=5.0):
    """
    直方图匹配进行色彩一致性校正
    
    参数:
        source_colors: 源颜色 (N, 3) 范围0-1
        reference_colors: 参考颜色 (M, 3)
        weights: 源颜色权重 (N,)，可选
        clip_percentile: 裁剪百分位（避免极端值影响）
    
    返回:
        matched_colors: 匹配后的颜色 (N, 3)
    """
    if len(source_colors) == 0 or len(reference_colors) == 0:
        return source_colors.copy()
    
    matched = np.zeros_like(source_colors)
    
    # 逐通道匹配
    for c in range(3):
        src_channel = source_colors[:, c]
        ref_channel = reference_colors[:, c]
        
        # 裁剪极端值
        src_min = np.percentile(src_channel, clip_percentile)
        src_max = np.percentile(src_channel, 100 - clip_percentile)
        ref_min = np.percentile(ref_channel, clip_percentile)
        ref_max = np.percentile(ref_channel, 100 - clip_percentile)
        
        # 线性拉伸匹配
        src_clipped = np.clip(src_channel, src_min, src_max)
        src_norm = (src_clipped - src_min) / (src_max - src_min + 1e-8)
        matched[:, c] = src_norm * (ref_max - ref_min) + ref_min
    
    # 裁剪到有效范围
    matched = np.clip(matched, 0, 1)
    
    # 如果提供了权重，对低权重区域保留原始颜色
    if weights is not None:
        alpha = np.clip(weights, 0, 1)[:, np.newaxis]
        matched = alpha * matched + (1 - alpha) * source_colors
    
    return matched


def color_transfer(source_colors, reference_colors, preserve_luminance=True):
    """
    色彩传递（Reinhard风格）
    
    参数:
        source_colors: 源颜色 (N, 3) 范围0-1
        reference_colors: 参考颜色 (M, 3)
        preserve_luminance: 是否保留亮度信息
    
    返回:
        transferred_colors: 传递后的颜色 (N, 3)
    """
    if len(source_colors) == 0 or len(reference_colors) == 0:
        return source_colors.copy()
    
    # 转换到LAB空间
    src_lab = rgb_to_lab(source_colors)
    ref_lab = rgb_to_lab(reference_colors)
    
    # 计算统计量
    src_mean = src_lab.mean(axis=0)
    src_std = src_lab.std(axis=0) + 1e-8
    ref_mean = ref_lab.mean(axis=0)
    ref_std = ref_lab.std(axis=0) + 1e-8
    
    # 色彩传递
    transferred_lab = src_lab.copy()
    
    if preserve_luminance:
        # 只传递色度（a*, b*通道）
        for c in [1, 2]:
            transferred_lab[:, c] = (src_lab[:, c] - src_mean[c]) / src_std[c]
            transferred_lab[:, c] = transferred_lab[:, c] * ref_std[c] + ref_mean[c]
    else:
        # 传递所有通道
        for c in range(3):
            transferred_lab[:, c] = (src_lab[:, c] - src_mean[c]) / src_std[c]
            transferred_lab[:, c] = transferred_lab[:, c] * ref_std[c] + ref_mean[c]
    
    # 转换回RGB
    transferred_colors = lab_to_rgb(transferred_lab)
    
    return transferred_colors


def color_correction(vertex_colors, reference_colors, weights=None,
                     method="histogram", **kwargs):
    """
    色彩一致性校正统一接口
    
    参数:
        vertex_colors: 顶点颜色 (N, 3)
        reference_colors: 参考颜色（如前一帧或全局平均）
        weights: 顶点权重 (N,)
        method: 校正方法 "histogram"/"transfer"
        **kwargs: 方法特定参数
    
    返回:
        corrected_colors: 校正后的颜色 (N, 3)
    """
    if method == "histogram":
        clip_percentile = kwargs.get('clip_percentile', 5.0)
        return histogram_match_colors(vertex_colors, reference_colors, 
                                     weights, clip_percentile)
    elif method == "transfer":
        preserve_luminance = kwargs.get('preserve_luminance', True)
        return color_transfer(vertex_colors, reference_colors, preserve_luminance)
    else:
        raise ValueError(f"不支持的校正方法: {method}")


def post_process_texture(vertex_colors, vertex_positions, weights,
                        config=None, reference_colors=None):
    """
    纹理后处理主流程（第8步优化）
    
    参数:
        vertex_colors: 顶点颜色 (N, 3)
        vertex_positions: 顶点3D坐标 (N, 3)
        weights: 顶点权重 (N,)
        config: 配置字典（可选，使用全局配置）
        reference_colors: 参考颜色（用于色彩校正）
    
    返回:
        processed_colors: 处理后的颜色 (N, 3)
        stats: 处理统计信息字典
    """
    if config is None:
        config = {
            'use_outlier_detection': USE_OUTLIER_DETECTION,
            'outlier_method': OUTLIER_DETECTION_METHOD,
            'outlier_zscore': OUTLIER_ZSCORE_THRESHOLD,
            'outlier_local_k': 8,
            'outlier_local_threshold': OUTLIER_LOCAL_THRESHOLD,
            'use_smooth': USE_EDGE_PRESERVING_SMOOTH,
            'smooth_method': SMOOTH_METHOD,
            'bilateral_sigma_spatial': BILATERAL_SIGMA_SPATIAL,
            'bilateral_sigma_color': BILATERAL_SIGMA_COLOR / 255.0,  # 转换到0-1
            'anisotropic_iterations': ANISOTROPIC_ITERATIONS,
            'anisotropic_kappa': ANISOTROPIC_KAPPA / 255.0,  # 转换到0-1
            'use_color_correction': USE_COLOR_CORRECTION,
            'color_method': COLOR_CORRECTION_METHOD,
            'histogram_clip': HISTOGRAM_MATCH_PERCENTILE,
            'transfer_preserve_lum': COLOR_TRANSFER_PRESERVE_LUMINANCE,
        }
    
    processed = vertex_colors.copy()
    stats = {}
    
    # 1. 异常值检测与移除
    if config['use_outlier_detection']:
        outlier_mask, outlier_count = detect_outliers(
            processed, vertex_positions, weights,
            method=config['outlier_method'],
            zscore_threshold=config['outlier_zscore'],
            local_k=config['outlier_local_k'],
            local_threshold=config['outlier_local_threshold']
        )
        
        # 移除异常值（将权重设为0）
        weights = weights.copy()
        weights[outlier_mask] = 0
        
        stats['outliers_detected'] = outlier_count
        stats['outliers_ratio'] = outlier_count / len(processed) if len(processed) > 0 else 0
    
    # 2. 边缘保持平滑
    if config['use_smooth']:
        if config['smooth_method'] == 'bilateral':
            processed = edge_preserving_smooth(
                processed, vertex_positions, weights,
                method='bilateral',
                sigma_spatial=config['bilateral_sigma_spatial'],
                sigma_color=config['bilateral_sigma_color'],
                k_neighbors=16
            )
        elif config['smooth_method'] == 'anisotropic':
            processed = edge_preserving_smooth(
                processed, vertex_positions, weights,
                method='anisotropic',
                iterations=config['anisotropic_iterations'],
                kappa=config['anisotropic_kappa'],
                gamma=0.1
            )
    
    # 3. 色彩一致性校正
    if config['use_color_correction'] and reference_colors is not None:
        if config['color_method'] == 'histogram':
            processed = color_correction(
                processed, reference_colors, weights,
                method='histogram',
                clip_percentile=config['histogram_clip']
            )
        elif config['color_method'] == 'transfer':
            processed = color_correction(
                processed, reference_colors, weights,
                method='transfer',
                preserve_luminance=config['transfer_preserve_lum']
            )
    
    return processed, stats


# ============================================================================
# 第3步优化：双三次插值函数
# ============================================================================
def cubic_kernel(x, a=-0.5):
    """
    双三次插值核函数（Catmull-Rom样条）
    
    参数:
        x: 距离（0-2范围）
        a: 插值参数（-0.75到-0.5，-0.5更锐利，-0.75更平滑）
    
    返回:
        权重值
    """
    x = np.abs(x)
    
    # 0 <= |x| < 1
    mask1 = x < 1
    # 1 <= |x| < 2
    mask2 = (x >= 1) & (x < 2)
    
    result = np.zeros_like(x)
    result[mask1] = (a + 2) * x[mask1]**3 - (a + 3) * x[mask1]**2 + 1
    result[mask2] = a * x[mask2]**3 - 5*a * x[mask2]**2 + 8*a * x[mask2] - 4*a
    
    return result

def bicubic_interpolate(img, u_f, v_f, a=-0.5):
    """
    双三次插值采样
    
    参数:
        img: 输入图像 (H, W, 3)
        u_f, v_f: 浮点坐标数组
        a: 插值参数
    
    返回:
        插值后的颜色 (N, 3)
    """
    H, W = img.shape[:2]
    
    # 获取16个邻域像素的坐标
    u_int = np.floor(u_f).astype(int)
    v_int = np.floor(v_f).astype(int)
    
    # 计算相对位置
    du = u_f - u_int
    dv = v_f - v_int
    
    # 初始化输出
    colors = np.zeros((len(u_f), 3), dtype=np.float32)
    
    # 16个邻域像素（4x4网格）
    for i in range(-1, 3):
        for j in range(-1, 3):
            # 邻域坐标
            u_neighbor = np.clip(u_int + i, 0, W - 1)
            v_neighbor = np.clip(v_int + j, 0, H - 1)
            
            # 计算权重
            weight_u = cubic_kernel(i - du, a)
            weight_v = cubic_kernel(j - dv, a)
            weight = weight_u * weight_v
            
            # 累加加权颜色
            colors += img[v_neighbor, u_neighbor] * weight[:, np.newaxis]
    
    return colors

# ============================================================================
# 第4步优化：智能视角选择与加权函数
# ============================================================================
def compute_view_angle_weight(normals, view_dirs, max_angle_deg=75.0):
    """
    计算视角质量权重（基于法向和视线夹角）
    
    参数:
        normals: 顶点法向量 (N, 3)，已归一化
        view_dirs: 视线方向 (N, 3)，已归一化，指向相机
        max_angle_deg: 最大有效视角（度）
    
    返回:
        视角权重 (N,)，范围[0, 1]，0度最优（权重1），90度最差（权重0）
    """
    # 计算法向和视线的夹角余弦值
    cos_angles = np.sum(normals * view_dirs, axis=1)
    cos_angles = np.clip(cos_angles, -1, 1)
    
    # 转换为角度
    angles_deg = np.degrees(np.arccos(cos_angles))
    
    # 使用平滑衰减函数：在max_angle之前缓慢衰减，之后快速衰减
    # 使用cos^2函数作为权重（更平滑的衰减）
    max_angle_rad = np.radians(max_angle_deg)
    
    # 超过最大角度的设为0
    weights = np.where(
        angles_deg <= max_angle_deg,
        (cos_angles ** 2),  # 0-max_angle范围内：cos^2衰减
        0.0                  # 超过max_angle：权重为0
    )
    
    return weights

def compute_distance_weight(distances, falloff=2.0):
    """
    计算距离权重（基于相机到表面距离）
    
    参数:
        distances: 相机到表面的距离 (N,)
        falloff: 衰减指数（越大衰减越快）
    
    返回:
        距离权重 (N,)，范围[0, 1]，近距离权重高，远距离权重低
    """
    # 归一化距离（相对于最小距离）
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist - min_dist < 1e-6:
        # 所有距离相同，返回均匀权重
        return np.ones_like(distances)
    
    # 归一化到[0, 1]
    normalized_dist = (distances - min_dist) / (max_dist - min_dist)
    
    # 使用指数衰减：weight = exp(-falloff * normalized_dist)
    # 或使用幂函数：weight = (1 - normalized_dist)^falloff
    weights = (1.0 - normalized_dist) ** falloff
    
    return weights

def compute_combined_weight(view_weights, dist_weights, quality_scores,
                           view_alpha=0.4, dist_alpha=0.3, quality_alpha=0.3):
    """
    计算综合权重（视角 + 距离 + 图像质量）
    
    参数:
        view_weights: 视角权重 (N,)
        dist_weights: 距离权重 (N,)
        quality_scores: 图像质量分数 (N,)，范围[0, 1]
        view_alpha: 视角权重系数
        dist_alpha: 距离权重系数
        quality_alpha: 图像质量权重系数
    
    返回:
        综合权重 (N,)，范围[0, 1]
    """
    # 归一化系数（确保总和为1）
    total_alpha = view_alpha + dist_alpha + quality_alpha
    view_alpha /= total_alpha
    dist_alpha /= total_alpha
    quality_alpha /= total_alpha
    
    # 加权组合
    combined = (view_alpha * view_weights +
                dist_alpha * dist_weights +
                quality_alpha * quality_scores)
    
    # 确保范围在[0, 1]
    combined = np.clip(combined, 0.0, 1.0)
    
    return combined

# ============================================================================
# 第5步优化：基于方差的接缝平滑 (Variance-based Seam Smoothing)
# ============================================================================

def compute_local_variance(mesh, vertex_colors, vertex_weights, k_neighbors=10):
    """
    计算每个顶点的局部颜色方差
    
    参数:
        mesh: Open3D三角网格
        vertex_colors: 顶点颜色累积值 (N, 3)
        vertex_weights: 顶点权重累积值 (N,)
        k_neighbors: 邻域顶点数量
    
    返回:
        局部方差值 (N,) 范围[0, +inf]，值越大表示接缝可能性越高
    """
    vertices_np = np.asarray(mesh.vertices)
    n_vertices = len(vertices_np)
    
    # 构建KD树用于邻域搜索
    from scipy.spatial import cKDTree
    kdtree = cKDTree(vertices_np)
    
    # 计算当前顶点颜色（加权平均）
    current_colors = np.zeros_like(vertex_colors)
    non_zero = vertex_weights > 0
    current_colors[non_zero] = vertex_colors[non_zero] / vertex_weights[non_zero, np.newaxis]
    
    # 计算局部方差
    variances = np.zeros(n_vertices)
    
    for i in range(n_vertices):
        if vertex_weights[i] == 0:
            variances[i] = 0.0  # 未着色顶点方差为0
            continue
        
        # 查询k个最近邻
        distances, indices = kdtree.query(vertices_np[i], k=k_neighbors+1)
        neighbor_indices = indices[1:]  # 排除自己
        
        # 只考虑已着色的邻居
        valid_neighbors = neighbor_indices[vertex_weights[neighbor_indices] > 0]
        
        if len(valid_neighbors) == 0:
            variances[i] = 0.0
            continue
        
        # 计算邻域颜色方差
        neighbor_colors = current_colors[valid_neighbors]
        color_diff = neighbor_colors - current_colors[i]
        variance = np.mean(np.sum(color_diff ** 2, axis=1))
        variances[i] = variance
    
    return variances


def detect_seam_regions(mesh, vertex_colors, vertex_weights, variance_threshold=0.01, k_neighbors=10):
    """
    检测可能存在接缝的区域
    
    参数:
        mesh: Open3D三角网格
        vertex_colors: 顶点颜色累积值 (N, 3)
        vertex_weights: 顶点权重累积值 (N,)
        variance_threshold: 方差阈值，超过此值认为是接缝
        k_neighbors: 邻域顶点数量
    
    返回:
        seam_mask: 接缝掩码 (N,) bool数组，True表示可能是接缝
        variances: 局部方差值 (N,)
    """
    variances = compute_local_variance(mesh, vertex_colors, vertex_weights, k_neighbors)
    seam_mask = variances > variance_threshold
    return seam_mask, variances


def apply_adaptive_smoothing(mesh, vertex_colors, vertex_weights, seam_mask, 
                             smoothing_strength=0.5, k_neighbors=10):
    """
    对接缝区域进行自适应平滑
    
    参数:
        mesh: Open3D三角网格
        vertex_colors: 顶点颜色累积值 (N, 3)
        vertex_weights: 顶点权重累积值 (N,)
        seam_mask: 接缝掩码 (N,) bool数组
        smoothing_strength: 平滑强度 [0, 1]，0=不平滑，1=完全平滑
        k_neighbors: 邻域顶点数量
    
    返回:
        平滑后的顶点颜色 (N, 3)
    """
    vertices_np = np.asarray(mesh.vertices)
    n_vertices = len(vertices_np)
    
    # 构建KD树
    from scipy.spatial import cKDTree
    kdtree = cKDTree(vertices_np)
    
    # 计算当前顶点颜色
    current_colors = np.zeros_like(vertex_colors)
    non_zero = vertex_weights > 0
    current_colors[non_zero] = vertex_colors[non_zero] / vertex_weights[non_zero, np.newaxis]
    
    # 创建平滑后的颜色副本
    smoothed_colors = current_colors.copy()
    
    # 只对接缝区域进行平滑
    seam_indices = np.where(seam_mask)[0]
    
    for i in seam_indices:
        if vertex_weights[i] == 0:
            continue
        
        # 查询k个最近邻
        distances, indices = kdtree.query(vertices_np[i], k=k_neighbors+1)
        neighbor_indices = indices[1:]  # 排除自己
        
        # 只考虑已着色的邻居
        valid_neighbors = neighbor_indices[vertex_weights[neighbor_indices] > 0]
        
        if len(valid_neighbors) == 0:
            continue
        
        # 基于距离的权重（距离越近权重越大）
        neighbor_distances = distances[1:][vertex_weights[neighbor_indices] > 0]
        neighbor_weights = 1.0 / (neighbor_distances + 1e-6)
        neighbor_weights = neighbor_weights / np.sum(neighbor_weights)
        
        # 计算加权平均颜色
        neighbor_colors = current_colors[valid_neighbors]
        
        # 第6步优化：可选在LAB空间混合（更好的感知一致性）
        if USE_LAB_COLOR_SPACE:
            # 在LAB空间混合邻域颜色
            all_colors = np.vstack([current_colors[i:i+1], neighbor_colors])
            all_weights = np.concatenate([[1 - smoothing_strength], neighbor_weights * smoothing_strength])
            all_weights = all_weights / np.sum(all_weights)  # 归一化
            
            avg_color = mix_colors_in_lab(
                all_colors, all_weights,
                l_weight=LAB_L_WEIGHT,
                normalize_l=LAB_NORMALIZE_L,
                l_clip_percentile=LAB_L_CLIP_PERCENTILE
            )
        else:
            # RGB空间简单加权平均
            avg_color = np.sum(neighbor_colors * neighbor_weights[:, np.newaxis], axis=0)
            avg_color = (1 - smoothing_strength) * current_colors[i] + smoothing_strength * avg_color
        
        smoothed_colors[i] = avg_color
    
    return smoothed_colors

# ============================================================================
# 第6步优化：LAB色彩空间转换函数
# ============================================================================
def rgb_to_lab(rgb):
    """
    将RGB颜色转换到LAB色彩空间
    
    LAB色彩空间优势：
    - L通道：亮度（0-100），与人眼感知一致
    - A通道：绿色到红色（-128到127）
    - B通道：蓝色到黄色（-128到127）
    - 分离亮度和色度，减少光照影响
    - 在LAB空间混合颜色更符合人眼感知
    
    参数:
        rgb: RGB颜色 (N, 3) 或 (3,)，范围[0, 1]
    
    返回:
        lab: LAB颜色 (N, 3) 或 (3,)
             L: [0, 100]
             A, B: [-128, 127]
    """
    # 确保输入是2D数组
    input_shape = rgb.shape
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, -1)
    
    # 1. RGB to XYZ（使用sRGB标准）
    # 先进行gamma校正（逆sRGB变换）
    rgb_linear = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )
    
    # RGB to XYZ转换矩阵（D65光源）
    transform_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = rgb_linear @ transform_matrix.T
    
    # 2. XYZ to LAB
    # D65标准光源白点
    xyz_n = np.array([0.95047, 1.00000, 1.08883])
    
    # 归一化XYZ
    xyz_norm = xyz / xyz_n
    
    # f函数（LAB转换的非线性部分）
    delta = 6.0 / 29.0
    def f(t):
        return np.where(
            t > delta3,
            t(1/3),
            t / (3 * delta2) + (4/29)
        )
    
    fx = f(xyz_norm[:, 0])
    fy = f(xyz_norm[:, 1])
    fz = f(xyz_norm[:, 2])
    
    # 计算LAB
    L = 116 * fy - 16
    A = 500 * (fx - fy)
    B = 200 * (fy - fz)
    
    lab = np.stack([L, A, B], axis=-1)
    
    # 恢复原始形状
    if len(input_shape) == 1:
        lab = lab.reshape(input_shape)
    
    return lab

def lab_to_rgb(lab):
    """
    将LAB颜色转换到RGB色彩空间
    
    参数:
        lab: LAB颜色 (N, 3) 或 (3,)
             L: [0, 100]
             A, B: [-128, 127]
    
    返回:
        rgb: RGB颜色 (N, 3) 或 (3,)，范围[0, 1]
    """
    # 确保输入是2D数组
    input_shape = lab.shape
    if lab.ndim == 1:
        lab = lab.reshape(1, -1)
    
    L = lab[:, 0]
    A = lab[:, 1]
    B = lab[:, 2]
    
    # 1. LAB to XYZ
    fy = (L + 16) / 116
    fx = A / 500 + fy
    fz = fy - B / 200
    
    # f逆函数
    delta = 6.0 / 29.0
    def f_inv(t):
        return np.where(
            t > delta,
            t3,
            3 * delta2 * (t - 4/29)
        )
    
    xyz_norm = np.stack([
        f_inv(fx),
        f_inv(fy),
        f_inv(fz)
    ], axis=-1)
    
    # D65标准光源白点
    xyz_n = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz_norm * xyz_n
    
    # 2. XYZ to RGB
    # XYZ to RGB转换矩阵（D65光源）
    transform_matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    rgb_linear = xyz @ transform_matrix.T
    
    # 3. 应用sRGB gamma校正
    rgb = np.where(
        rgb_linear > 0.0031308,
        1.055 * (rgb_linear  (1/2.4)) - 0.055,
        12.92 * rgb_linear
    )
    
    # 裁剪到[0, 1]范围
    rgb = np.clip(rgb, 0, 1)
    
    # 恢复原始形状
    if len(input_shape) == 1:
        rgb = rgb.reshape(input_shape)
    
    return rgb

def mix_colors_in_lab(colors, weights, l_weight=0.5, normalize_l=True, l_clip_percentile=2.0):
    """
    在LAB色彩空间中混合颜色（减少光照影响）
    
    核心思想：
    - L通道（亮度）可选归一化，减少不同光照条件的影响
    - A/B通道（色度）保留完整信息
    - 加权混合后转回RGB
    
    参数:
        colors: RGB颜色数组 (N, 3)，范围[0, 1]
        weights: 权重 (N,)，已归一化
        l_weight: L通道权重（0-1），越高越保留亮度差异
        normalize_l: 是否归一化L通道（减少光照影响）
        l_clip_percentile: L通道裁剪百分位（避免极端值）
    
    返回:
        mixed_rgb: 混合后的RGB颜色 (3,)，范围[0, 1]
    """
    if len(colors) == 0:
        return np.array([0.5, 0.5, 0.5])
    
    # 转换到LAB空间
    lab_colors = rgb_to_lab(colors)  # (N, 3)
    
    L = lab_colors[:, 0]  # (N,)
    A = lab_colors[:, 1]
    B = lab_colors[:, 2]
    
    # 可选：归一化L通道（减少光照影响）
    if normalize_l and len(L) > 1:
        # 使用百分位裁剪避免极端值
        L_min = np.percentile(L, l_clip_percentile)
        L_max = np.percentile(L, 100 - l_clip_percentile)
        
        if L_max - L_min > 1e-6:
            L_normalized = (L - L_min) / (L_max - L_min) * 100
            L_normalized = np.clip(L_normalized, 0, 100)
            
            # 混合原始L和归一化L
            L = l_weight * L + (1 - l_weight) * L_normalized
    
    # 加权平均（在LAB空间）
    L_mixed = np.sum(L * weights)
    A_mixed = np.sum(A * weights)
    B_mixed = np.sum(B * weights)
    
    # 转回RGB
    lab_mixed = np.array([L_mixed, A_mixed, B_mixed])
    rgb_mixed = lab_to_rgb(lab_mixed)
    
    return rgb_mixed

# ============================================================================
# 车辆检测与去除函数
# ============================================================================
def detect_vehicles(depth_measured, depth_rendered, threshold=0.5, min_depth=0.1):
    """
    基于深度差异检测车辆位置（简单版本，向后兼容）
    
    原理：mesh只包含地面，depth图包含车辆
    当 depth_measured >> depth_rendered 时，说明有车辆遮挡
    
    参数:
        depth_measured: depth图的深度 (H, W)
        depth_rendered: mesh渲染深度 (H, W) 
        threshold: 高度差阈值（米），车辆高度
        min_depth: 最小有效深度
    
    返回:
        vehicle_mask: 布尔数组 (H, W)，True=车辆位置
    """
    # 计算深度差异（正值 = 有物体在mesh前面）
    depth_diff = depth_measured - depth_rendered
    
    # 车辆mask：深度差异大于阈值，且两个深度都有效
    vehicle_mask = (depth_diff > threshold) & \
                   (depth_measured > min_depth) & \
                   (depth_rendered > min_depth)
    
    return vehicle_mask


def detect_vehicles_multimodal(depth_obs, depth_tsdf, normals=None, points_3d=None, 
                                config=None):
    """
    四模态车辆检测（利用OSM-TSDF先验）
    
    原理：综合四种几何线索进行车辆检测
    1. 法向检测：地面法向量接近垂直
    2. 高度过滤：车辆位于地面之上特定高度范围
    3. 深度不连续：车辆边缘深度梯度大
    4. TSDF深度一致性：观测深度 < TSDF深度（前景遮挡）
    
    参数:
        depth_obs: 观测深度图 (H, W)
        depth_tsdf: TSDF渲染深度 (H, W)
        normals: 表面法向量 (H, W, 3)，用于法向检测
        points_3d: 三维点云 (H, W, 3)，用于高度过滤
        config: 配置字典
    
    返回:
        vehicle_mask: 布尔数组 (H, W)，True=车辆位置
        cue_masks: 各线索的独立检测结果（调试用）
    """
    if config is None:
        config = {
            'use_ground_normal': USE_GROUND_NORMAL,
            'ground_normal_threshold': GROUND_NORMAL_THRESHOLD,
            'use_height_filter': USE_HEIGHT_FILTER,
            'height_min': VEHICLE_HEIGHT_MIN,
            'height_max': VEHICLE_HEIGHT_MAX,
            'use_depth_discontinuity': USE_DEPTH_DISCONTINUITY,
            'gradient_threshold': DEPTH_GRADIENT_THRESHOLD,
            'use_depth_consistency': USE_TSDF_DEPTH_CONSISTENCY,
            'depth_diff_threshold': DEPTH_DIFF_THRESHOLD,
            'depth_noise_tolerance': DEPTH_NOISE_TOLERANCE,
            'require_all_cues': REQUIRE_ALL_CUES,
            'min_depth': MIN_DEPTH,
            'max_depth': MAX_DEPTH,
        }
    
    H, W = depth_obs.shape
    cue_masks = {}
    
    # ========== 线索1: 地面法向检测 ==========
    if config['use_ground_normal'] and normals is not None:
        # 法向量的Z分量（垂直分量）
        normal_z = normals[:, :, 2]
        # 接近垂直向上的为地面
        ground_mask = normal_z > config['ground_normal_threshold']
        cue_masks['ground_normal'] = ground_mask
    else:
        ground_mask = np.ones((H, W), dtype=bool)
        cue_masks['ground_normal'] = None
    
    # ========== 线索2: 车辆高度范围过滤 ==========
    if config['use_height_filter'] and points_3d is not None:
        # 获取点的Z坐标（高度）
        heights = points_3d[:, :, 2]
        # 估计地面高度（取最小值的中位数）
        valid_heights = heights[depth_obs > 0]
        if len(valid_heights) > 100:
            ground_height = np.percentile(valid_heights, 10)
        else:
            ground_height = 0.0
        
        # 相对高度
        relative_height = heights - ground_height
        # 车辆高度范围
        height_mask = (relative_height >= config['height_min']) & \
                      (relative_height <= config['height_max'])
        cue_masks['height_filter'] = height_mask
    else:
        height_mask = np.ones((H, W), dtype=bool)
        cue_masks['height_filter'] = None
    
    # ========== 线索3: 深度不连续性检测 ==========
    if config['use_depth_discontinuity']:
        # 计算深度梯度（Sobel算子）
        grad_x = cv2.Sobel(depth_obs, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_obs, cv2.CV_64F, 0, 1, ksize=3)
        depth_gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 深度突变区域（车辆边缘）
        discontinuity_mask = depth_gradient > config['gradient_threshold']
        cue_masks['depth_discontinuity'] = discontinuity_mask
    else:
        discontinuity_mask = np.ones((H, W), dtype=bool)
        cue_masks['depth_discontinuity'] = None
    
    # ========== 线索4: TSDF深度一致性检测（新增！）==========
    if config['use_depth_consistency']:
        # 深度差异：TSDF理论深度 - 观测深度
        depth_diff = depth_tsdf - depth_obs
        
        # 当观测深度显著小于TSDF深度时，说明有前景遮挡（车辆）
        # D_TSDF - D_obs > threshold → 前景物体
        consistency_mask = depth_diff > config['depth_diff_threshold']
        
        # 深度有效性检查
        valid_obs = (depth_obs > config['min_depth']) & (depth_obs < config['max_depth'])
        valid_tsdf = (depth_tsdf > config['min_depth']) & (depth_tsdf < config['max_depth'])
        valid_mask = valid_obs & valid_tsdf
        
        # 综合判断
        consistency_mask = consistency_mask & valid_mask
        cue_masks['depth_consistency'] = consistency_mask
    else:
        consistency_mask = np.ones((H, W), dtype=bool)
        cue_masks['depth_consistency'] = None
    
    # ========== 多模态融合 ==========
    if config['require_all_cues']:
        # AND逻辑：所有线索都满足
        vehicle_mask = ground_mask & height_mask & discontinuity_mask & consistency_mask
    else:
        # OR逻辑：任意线索满足
        vehicle_mask = ground_mask | height_mask | discontinuity_mask | consistency_mask
    
    return vehicle_mask, cue_masks

def refine_vehicle_mask(mask, dilation=5):
    """
    形态学处理：填充车辆内部空洞、平滑边缘、扩大覆盖范围
    
    参数:
        mask: 布尔数组 (H, W)
        dilation: 膨胀半径（像素）
    
    返回:
        refined_mask: 处理后的布尔数组 (H, W)
    """
    if not np.any(mask):
        return mask
    
    # 转换为uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # 创建椭圆形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (dilation*2+1, dilation*2+1))
    
    # 闭运算：填充车辆内部空洞
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # 膨胀：扩大mask，确保完全覆盖车辆（包括边缘）
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    return mask_uint8.astype(bool)

def fill_empty_vertices(vertex_colors, vertex_weights, vertices, k=8):
    """
    使用K近邻插值填充权重为0的顶点（完全未投影的区域）
    
    应用场景：车辆遮挡导致某些地面区域没有纹理
    
    参数:
        vertex_colors: 顶点颜色数组 (N, 3)
        vertex_weights: 顶点权重数组 (N,)
        vertices: 顶点坐标数组 (N, 3)
        k: K近邻数量
    
    返回:
        filled_colors: 填充后的颜色数组 (N, 3)
    """
    from scipy.spatial import KDTree
    
    # 找到空白顶点（权重为0）
    empty_mask = (vertex_weights == 0)
    n_empty = np.sum(empty_mask)
    
    if n_empty == 0:
        return vertex_colors
    
    print(f"      📍 检测到 {n_empty} 个空白顶点，使用K近邻填充...")
    
    # 找到有颜色的顶点
    valid_mask = (vertex_weights > 0)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        print(f"      ⚠️  没有有效顶点，无法填充")
        return vertex_colors
    
    # 构建KD树（所有有颜色的顶点）
    valid_vertices = vertices[valid_mask]
    valid_colors = vertex_colors[valid_mask]
    kdtree = KDTree(valid_vertices)
    
    # 查询空白顶点的K近邻
    empty_vertices = vertices[empty_mask]
    k_actual = min(k, n_valid)  # 确保k不超过有效顶点数
    distances, indices = kdtree.query(empty_vertices, k=k_actual)
    
    # 加权平均（距离越近权重越大）
    # 使用反距离加权 (Inverse Distance Weighting)
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # 插值颜色
    if k_actual == 1:
        # 如果只有1个近邻，直接使用
        interpolated_colors = valid_colors[indices.flatten()]
    else:
        # 多个近邻，加权平均
        interpolated_colors = (valid_colors[indices] * weights[:, :, np.newaxis]).sum(axis=1)
    
    # 填充
    filled_colors = vertex_colors.copy()
    filled_colors[empty_mask] = interpolated_colors
    
    print(f"      ✓ 填充完成")
    
    return filled_colors

# ============================================================================
# 自适应深度检测函数
# ============================================================================
def adaptive_depth_test(depth_rendered, depth_measured, vertex_normals_world, 
                       up_vector=np.array([0, 0, 1]),
                       floor_normal_threshold=0.7,
                       floor_depth_front=0.3, floor_depth_back=0.08,
                       wall_depth_front=0.5, wall_depth_back=0.15,
                       min_depth=0.1, max_depth=20.0):
    """
    自适应深度一致性检测：根据表面类型（地面/墙壁）和方向（前/后）使用不同阈值
    
    参数:
        depth_rendered: 渲染深度（mesh顶点到相机的距离）(N,)
        depth_measured: 测量深度（depth图中的深度值）(N,)
        vertex_normals_world: 世界坐标系下的顶点法向量 (N, 3)
        up_vector: 世界坐标系的向上方向 (3,)
        floor_normal_threshold: 判断地面的法向量阈值（cos值，默认0.7约45°）
        floor_depth_front: 地面在前时的深度容差（米）
        floor_depth_back: 地面在后时的深度容差（米）- 严格防穿透
        wall_depth_front: 墙壁在前时的深度容差（米）
        wall_depth_back: 墙壁在后时的深度容差（米）
        min_depth: 最小有效深度
        max_depth: 最大有效深度
    
    返回:
        depth_consistency_mask: 布尔掩码 (N,)，True表示通过深度检测
    """
    # 1. 计算法向量与向上方向的点积
    up_vector_norm = up_vector / np.linalg.norm(up_vector)
    normal_dot_up = np.dot(vertex_normals_world, up_vector_norm)
    
    # 2. 分类表面类型：地面 vs 墙壁
    is_floor = normal_dot_up > floor_normal_threshold
    is_wall = ~is_floor
    
    # 3. 计算深度差异（带符号）
    # 正值：表面在depth后面（可能被遮挡/穿透）
    # 负值：表面在depth前面
    depth_diff = depth_rendered - depth_measured
    is_behind = depth_diff >= 0
    
    # 4. 根据表面类型和方向应用不同阈值
    depth_mask = np.zeros(len(depth_diff), dtype=bool)
    
    # 地面：在前时允许一定容差，在后时严格限制（防穿透污染）
    floor_front_mask = is_floor & ~is_behind
    floor_back_mask = is_floor & is_behind
    depth_mask[floor_front_mask] = depth_diff[floor_front_mask] > -floor_depth_front
    depth_mask[floor_back_mask] = depth_diff[floor_back_mask] < floor_depth_back
    
    # 墙壁：在前时较宽松（避免残缺），在后时适度限制（防穿透）
    wall_front_mask = is_wall & ~is_behind
    wall_back_mask = is_wall & is_behind
    depth_mask[wall_front_mask] = depth_diff[wall_front_mask] > -wall_depth_front
    depth_mask[wall_back_mask] = depth_diff[wall_back_mask] < wall_depth_back
    
    # 5. 结合深度范围检查
    depth_consistency_mask = depth_mask & (depth_measured > min_depth) & (depth_measured < max_depth)
    
    return depth_consistency_mask

# ============================================================================
# 旋转配置管理函数
# ============================================================================
def save_rotation_config(rx, ry, rz, config_file=ROTATION_CONFIG_FILE):
    """保存旋转配置到JSON文件"""
    import json
    config = {
        'rotation': {
            'x': float(rx),
            'y': float(ry),
            'z': float(rz)
        },
        'timestamp': str(datetime.now())
    }
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 旋转配置已保存到: {config_file}")

def load_rotation_config(config_file=ROTATION_CONFIG_FILE):
    """从JSON文件加载旋转配置"""
    import json
    config_path = Path(config_file)
    if not config_path.exists():
        return None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        rotation = config.get('rotation', {})
        rx = rotation.get('x', 0)
        ry = rotation.get('y', 0)
        rz = rotation.get('z', 0)
        timestamp = config.get('timestamp', '未知')
        print(f"📂 从配置文件加载旋转:")
        print(f"   文件: {config_file}")
        print(f"   旋转: X={rx}°, Y={ry}°, Z={rz}°")
        print(f"   时间: {timestamp}")
        return (rx, ry, rz)
    except Exception as e:
        print(f"⚠️  加载配置文件失败: {e}")
        return None

# ============================================================================
# 第一阶段：加载并验证轨迹
# ============================================================================
print(f"\n" + "="*70)
print(f"阶段 1：轨迹验证")
print(f"="*70)

print(f"\n[1/3] 读取首帧世界位姿")
T_world_from_first = np.eye(4, dtype=np.float32)
with open(first_pose_txt, 'r') as f:
    lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines[:4]):
        vals = [float(x) for x in line.split()]
        T_world_from_first[i, :] = vals

first_position = T_world_from_first[:3, 3]
print(f"   首帧世界位置: [{first_position[0]:.2f}, {first_position[1]:.2f}, {first_position[2]:.2f}]")

print(f"\n[2/3] 读取相对位姿文件")
pose_files = sorted(Path(pose_dir).glob("*.txt"), key=lambda x: int(x.stem))
print(f"   找到 {len(pose_files)} 个位姿文件")

# 存储世界位姿
world_poses = []
world_positions = []

for i, pose_file in enumerate(pose_files):
    T_first_from_current = np.eye(4, dtype=np.float32)
    with open(pose_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        for j, line in enumerate(lines[:4]):
            vals = [float(x) for x in line.split()]
            T_first_from_current[j, :] = vals
    
    T_world_from_current = T_world_from_first @ T_first_from_current
    world_poses.append(T_world_from_current)
    world_positions.append(T_world_from_current[:3, 3])

world_positions = np.array(world_positions)

# 固定摄像头高度（如果启用）
if FIX_CAMERA_HEIGHT:
    if FIXED_HEIGHT is None:
        # 使用首帧高度作为固定高度
        fixed_z = world_positions[0, 2]
    else:
        fixed_z = FIXED_HEIGHT
    
    print(f"\n   🔒 固定摄像头高度: Z = {fixed_z:.3f}")
    
    # 更新所有位姿的Z坐标
    for i in range(len(world_poses)):
        world_poses[i][2, 3] = fixed_z
        world_positions[i, 2] = fixed_z

# 强制相机水平（如果启用）
if FORCE_CAMERA_HORIZONTAL:
    print(f"\n   📐 强制相机水平（消除roll和pitch）")
    corrected_count = 0
    
    for i in range(len(world_poses)):
        # 提取当前旋转矩阵
        R_current = world_poses[i][:3, :3]
        
        # 提取yaw角（绕Z轴的旋转）
        # 从旋转矩阵中提取朝向向量（前向向量，相机的Z轴方向）
        forward = R_current[:, 2]  # 相机的Z轴在世界坐标系中的方向
        
        # 将forward向量投影到XY平面（水平面）
        forward_horizontal = np.array([forward[0], forward[1], 0])
        
        # 如果forward几乎垂直（forward_horizontal长度接近0），保持原样
        if np.linalg.norm(forward_horizontal) < 1e-6:
            continue
        
        # 归一化水平朝向
        forward_horizontal = forward_horizontal / np.linalg.norm(forward_horizontal)
        
        # 构造新的旋转矩阵（强制水平）
        # 相机坐标系：X右，Y下，Z前
        # 世界坐标系：Z上
        up_world = np.array([0, 0, 1])  # 世界向上方向
        right = np.cross(forward_horizontal, up_world)  # X = Z × up
        right = right / np.linalg.norm(right)
        down = np.cross(forward_horizontal, right)  # Y = Z × X
        
        # 新的旋转矩阵（列向量是相机坐标系的轴在世界坐标系中的表示）
        R_new = np.column_stack([right, down, forward_horizontal])
        
        # 更新位姿
        world_poses[i][:3, :3] = R_new
        corrected_count += 1
    
    print(f"      ✓ 已校正 {corrected_count}/{len(world_poses)} 个位姿")

print(f"   位姿统计:")
print(f"      总数: {len(world_poses)}")
print(f"      X范围: [{world_positions[:,0].min():.2f}, {world_positions[:,0].max():.2f}]")
print(f"      Y范围: [{world_positions[:,1].min():.2f}, {world_positions[:,1].max():.2f}]")
print(f"      Z范围: [{world_positions[:,2].min():.2f}, {world_positions[:,2].max():.2f}]")

print(f"\n[3/3] 加载网格")
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
print(f"   顶点数: {len(vertices):,}")
print(f"   网格范围:")
print(f"      X: [{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}]")
print(f"      Y: [{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}]")
print(f"      Z: [{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")

mesh.paint_uniform_color([0.8, 0.8, 0.8])

# ============================================================================
# GPU/CPU辅助函数
# ============================================================================
def to_gpu(array):
    """将NumPy数组传输到GPU（如果启用GPU）"""
    if USE_GPU and array is not None:
        try:
            return cp.asarray(array)
        except Exception as e:
            print(f"⚠️  GPU传输失败，使用CPU: {e}")
            return array
    return array

def to_cpu(array):
    """将数组传回CPU（如果是GPU数组）"""
    if USE_GPU and hasattr(array, 'get'):
        try:
            return cp.asnumpy(array)
        except Exception as e:
            print(f"⚠️  GPU下载失败: {e}")
            return array
    return array

def get_array_module(array):
    """获取数组对应的模块（numpy或cupy）"""
    if USE_GPU and hasattr(array, 'get'):
        return cp
    return np

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if USE_GPU:
        try:
            mempool = cp.get_default_memory_pool()
            used = mempool.used_bytes() / 10243
            total = cp.cuda.Device(GPU_DEVICE_ID).mem_info[1] / 10243
            return used, total
        except:
            return 0, 0
    return 0, 0

# ============================================================================
# 创建轨迹可视化几何体
# ============================================================================
def create_arrows_for_poses(poses_list, indices):
    """创建箭头几何体"""
    arrows = []
    for idx in indices:
        T = poses_list[idx]
        position = T[:3, 3]
        z_axis = T[:3, 2]  # Z轴方向（相机朝向）
        
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.1 * ARROW_SCALE,
            cone_radius=0.2 * ARROW_SCALE,
            cylinder_height=1.5 * ARROW_LENGTH,
            cone_height=0.5 * ARROW_LENGTH
        )
        
        # 计算旋转：从Z轴旋转到z_axis方向
        default_z = np.array([0, 0, 1])
        forward_normalized = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        
        if np.abs(np.dot(forward_normalized, default_z)) > 0.999:
            if np.dot(forward_normalized, default_z) < 0:
                arrow_rotation = R.from_euler('x', 180, degrees=True).as_matrix()
            else:
                arrow_rotation = np.eye(3)
        else:
            v = np.cross(default_z, forward_normalized)
            s = np.linalg.norm(v)
            c = np.dot(default_z, forward_normalized)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            arrow_rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))
        
        arrow.rotate(arrow_rotation, center=[0, 0, 0])
        arrow.translate(position)
        arrow.paint_uniform_color([0, 0.5, 1])
        arrows.append(arrow)
    return arrows

print(f"\n创建轨迹可视化...")
geometries = [mesh]

# 轨迹线
trajectory_points = world_positions
trajectory_lines = [[i, i+1] for i in range(len(trajectory_points)-1)]
trajectory_lineset = o3d.geometry.LineSet()
trajectory_lineset.points = o3d.utility.Vector3dVector(trajectory_points)
trajectory_lineset.lines = o3d.utility.Vector2iVector(trajectory_lines)
trajectory_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in trajectory_lines])
geometries.append(trajectory_lineset)

# 起点标记
start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
start_sphere.translate(world_positions[0])
start_sphere.paint_uniform_color([0, 1, 0])
geometries.append(start_sphere)

# 终点标记
end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
end_sphere.translate(world_positions[-1])
end_sphere.paint_uniform_color([1, 0, 0])
geometries.append(end_sphere)

# 方向箭头
sampled_indices = list(range(0, len(world_poses), SAMPLE_RATE))
arrows = create_arrows_for_poses(world_poses, sampled_indices)
geometries.extend(arrows)

# ============================================================================
# 旋转控制器（从 verify_trajectory_from_txt.py 移植）
# ============================================================================
class RotationController:
    def __init__(self, world_poses, world_positions):
        self.world_poses_original = [p.copy() for p in world_poses]
        self.world_positions_original = world_positions.copy()
        self.world_poses = world_poses
        self.world_positions = world_positions
        self.rx = self.ry = self.rz = 0
        self.rotation_center = world_positions[0].copy()
        
    def apply_rotation(self):
        rotation = R.from_euler('xyz', [self.rx, self.ry, self.rz], degrees=True).as_matrix()
        for i in range(len(self.world_poses_original)):
            T_orig = self.world_poses_original[i]
            pos_orig = T_orig[:3, 3]
            pos_new = rotation @ (pos_orig - self.rotation_center) + self.rotation_center
            rot_orig = T_orig[:3, :3]
            rot_new = rotation @ rot_orig
            self.world_poses[i][:3, :3] = rot_new
            self.world_poses[i][:3, 3] = pos_new
            self.world_positions[i] = pos_new
        print(f"\n🔄 应用旋转: X={self.rx}°, Y={self.ry}°, Z={self.rz}°")
        return True
    
    def rotate_x(self, degrees):
        self.rx = (self.rx + degrees) % 360
        return self.apply_rotation()
    
    def rotate_y(self, degrees):
        self.ry = (self.ry + degrees) % 360
        return self.apply_rotation()
    
    def rotate_z(self, degrees):
        self.rz = (self.rz + degrees) % 360
        return self.apply_rotation()
    
    def reset(self):
        self.rx = self.ry = self.rz = 0
        for i in range(len(self.world_poses_original)):
            self.world_poses[i][:] = self.world_poses_original[i]
            self.world_positions[i][:] = self.world_positions_original[i]
        print(f"\n🔄 重置旋转")
        return True

class OrientationController:
    def __init__(self, world_poses):
        self.world_poses = world_poses
        self.orientation_base = [pose[:3, :3].copy() for pose in world_poses]
        self.ox = self.oy = self.oz = 0
        self.enabled = False
        
    def lock_positions(self):
        self.orientation_base = [pose[:3, :3].copy() for pose in self.world_poses]
        self.ox = self.oy = self.oz = 0
        self.enabled = True
        print(f"\n🔒 已锁定坐标位置，进入朝向调整模式")
        return True
    
    def apply_orientation_rotation(self):
        if not self.enabled:
            print(f"⚠️  请先按 [L] 锁定坐标位置")
            return False
        
        rx_rad = np.radians(self.ox)
        ry_rad = np.radians(self.oy)
        rz_rad = np.radians(self.oz)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx_rad), -np.sin(rx_rad)],
            [0, np.sin(rx_rad), np.cos(rx_rad)]
        ])
        Ry = np.array([
            [np.cos(ry_rad), 0, np.sin(ry_rad)],
            [0, 1, 0],
            [-np.sin(ry_rad), 0, np.cos(ry_rad)]
        ])
        Rz = np.array([
            [np.cos(rz_rad), -np.sin(rz_rad), 0],
            [np.sin(rz_rad), np.cos(rz_rad), 0],
            [0, 0, 1]
        ])
        
        rotation = Rz @ Ry @ Rx
        for i in range(len(self.world_poses)):
            rot_base = self.orientation_base[i]
            rot_new = rotation @ rot_base
            self.world_poses[i][:3, :3] = rot_new
        
        print(f"\n🔄 应用朝向旋转: X={self.ox}°, Y={self.oy}°, Z={self.oz}°")
        return True
    
    def rotate_orientation_x(self, degrees):
        if not self.enabled:
            print(f"⚠️  请先按 [L] 锁定坐标位置")
            return False
        self.ox = (self.ox + degrees) % 360
        return self.apply_orientation_rotation()
    
    def rotate_orientation_y(self, degrees):
        if not self.enabled:
            print(f"⚠️  请先按 [L] 锁定坐标位置")
            return False
        self.oy = (self.oy + degrees) % 360
        return self.apply_orientation_rotation()
    
    def rotate_orientation_z(self, degrees):
        if not self.enabled:
            print(f"⚠️  请先按 [L] 锁定坐标位置")
            return False
        self.oz = (self.oz + degrees) % 360
        return self.apply_orientation_rotation()
    
    def reset_orientation(self):
        if not self.enabled:
            print(f"⚠️  请先按 [L] 锁定坐标位置")
            return False
        self.ox = self.oy = self.oz = 0
        for i in range(len(self.world_poses)):
            self.world_poses[i][:3, :3] = self.orientation_base[i]
        print(f"\n🔄 重置朝向")
        return True

rotation_controller = RotationController(world_poses, world_positions)
orientation_controller = OrientationController(world_poses)

# ============================================================================
# 可视化更新函数
# ============================================================================
def update_all_geometries_on_vis_thread(vis):
    """更新几何体"""
    # 更新轨迹线
    trajectory_points = rotation_controller.world_positions
    trajectory_lineset.points = o3d.utility.Vector3dVector(trajectory_points)
    vis.update_geometry(trajectory_lineset)
    
    # 更新起点球
    start_sphere.clear()
    start_sphere_temp = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    start_sphere_temp.translate(rotation_controller.world_positions[0])
    start_sphere_temp.paint_uniform_color([0, 1, 0])
    start_sphere.vertices = start_sphere_temp.vertices
    start_sphere.triangles = start_sphere_temp.triangles
    start_sphere.vertex_colors = start_sphere_temp.vertex_colors
    vis.update_geometry(start_sphere)
    
    # 更新终点球
    end_sphere.clear()
    end_sphere_temp = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    end_sphere_temp.translate(rotation_controller.world_positions[-1])
    end_sphere_temp.paint_uniform_color([1, 0, 0])
    end_sphere.vertices = end_sphere_temp.vertices
    end_sphere.triangles = end_sphere_temp.triangles
    end_sphere.vertex_colors = end_sphere_temp.vertex_colors
    vis.update_geometry(end_sphere)
    
    # 更新箭头
    arrow_geometries = geometries[4:]
    for i, idx in enumerate(sampled_indices):
        if i < len(arrow_geometries):
            arrow = arrow_geometries[i]
            arrow.clear()
            
            pose = rotation_controller.world_poses[idx]
            position = pose[:3, 3]
            z_axis = pose[:3, 2]
            
            arrow_temp = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.1 * ARROW_SCALE,
                cone_radius=0.2 * ARROW_SCALE,
                cylinder_height=1.5 * ARROW_LENGTH,
                cone_height=0.5 * ARROW_LENGTH
            )
            
            default_z = np.array([0, 0, 1])
            forward_normalized = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            
            if np.abs(np.dot(forward_normalized, default_z)) > 0.999:
                if np.dot(forward_normalized, default_z) < 0:
                    arrow_rotation = R.from_euler('x', 180, degrees=True).as_matrix()
                else:
                    arrow_rotation = np.eye(3)
            else:
                v = np.cross(default_z, forward_normalized)
                s = np.linalg.norm(v)
                c = np.dot(default_z, forward_normalized)
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                arrow_rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))
            
            arrow_temp.rotate(arrow_rotation, center=[0, 0, 0])
            arrow_temp.translate(position)
            arrow_temp.paint_uniform_color([0, 0.5, 1])
            
            arrow.vertices = arrow_temp.vertices
            arrow.triangles = arrow_temp.triangles
            arrow.vertex_colors = arrow_temp.vertex_colors
            vis.update_geometry(arrow)
    
    vis.update_renderer()

# ============================================================================
# 启动轨迹验证窗口
# ============================================================================
print(f"\n" + "="*70)
print(f"启动轨迹验证窗口...")
print(f"="*70)

# 检查是否有已保存的配置
saved_config = load_rotation_config() if AUTO_LOAD_ROTATION else None
if saved_config:
    print(f"\n💡 提示: 检测到已保存的旋转配置，启动后可直接使用")

print(f"\n🎮 第一阶段：坐标位置旋转")
print(f"  [1/2] 绕X轴: +90° / -90°")
print(f"  [3/4] 绕Y轴: +90° / -90°")
print(f"  [5/6] 绕Z轴: +90° / -90°")
print(f"  [R]   重置旋转")
print(f"\n🎯 第二阶段：相机朝向调整")
print(f"  [L]   锁定坐标位置")
print(f"  [7/8] 朝向X轴: +90° / -90°")
print(f"  [9/0] 朝向Y轴: +90° / -90°")
print(f"  [U/I] 朝向Z轴: +90° / -90°")
print(f"  [O]   重置朝向")
print(f"\n💾 配置管理:")
print(f"  [S]     保存当前旋转配置")
print(f"  [ENTER] 保存配置并开始纹理化")
print(f"\n基本操作:")
print(f"  - 鼠标拖动: 旋转视角")
print(f"  - 滚轮: 缩放")
print(f"  - Q/ESC: 退出\n")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="轨迹验证 - 按ENTER确认后开始纹理化", width=1600, height=900)

for geom in geometries:
    vis.add_geometry(geom)

render_option = vis.get_render_option()
render_option.mesh_show_back_face = True
render_option.light_on = True

# 命令队列
command_queue = deque()
queue_lock = threading.Lock()
should_quit = [False]
start_texturing = [False]

def enqueue_command(cmd: str):
    with queue_lock:
        command_queue.append(cmd)

def dequeue_commands():
    with queue_lock:
        items = list(command_queue)
        command_queue.clear()
        return items

# 键盘映射
key_to_action = {
    ord('1'): lambda: rotation_controller.rotate_x(90),
    ord('2'): lambda: rotation_controller.rotate_x(-90),
    ord('3'): lambda: rotation_controller.rotate_y(90),
    ord('4'): lambda: rotation_controller.rotate_y(-90),
    ord('5'): lambda: rotation_controller.rotate_z(90),
    ord('6'): lambda: rotation_controller.rotate_z(-90),
    ord('R'): lambda: rotation_controller.reset(),
    ord('r'): lambda: rotation_controller.reset(),
    ord('L'): lambda: orientation_controller.lock_positions(),
    ord('l'): lambda: orientation_controller.lock_positions(),
    ord('7'): lambda: orientation_controller.rotate_orientation_x(90),
    ord('8'): lambda: orientation_controller.rotate_orientation_x(-90),
    ord('9'): lambda: orientation_controller.rotate_orientation_y(90),
    ord('0'): lambda: orientation_controller.rotate_orientation_y(-90),
    ord('U'): lambda: orientation_controller.rotate_orientation_z(90),
    ord('u'): lambda: orientation_controller.rotate_orientation_z(90),
    ord('I'): lambda: orientation_controller.rotate_orientation_z(-90),
    ord('i'): lambda: orientation_controller.rotate_orientation_z(-90),
    ord('O'): lambda: orientation_controller.reset_orientation(),
    ord('o'): lambda: orientation_controller.reset_orientation(),
}

def animation_callback(vis):
    if should_quit[0] or start_texturing[0]:
        return False

    cmds = dequeue_commands()
    if not cmds:
        return True

    updated = False
    for cmd in cmds:
        if cmd in ('Q', 'QUIT', 'EXIT'):
            print("退出...")
            should_quit[0] = True
            return False
        elif cmd == 'START':
            print("\n✅ 轨迹确认，准备开始纹理化...")
            start_texturing[0] = True
            return False
        
        for ch in cmd:
            key_code = ord(ch)
            if key_code in key_to_action:
                result = key_to_action[key_code]()
                if result:
                    updated = True

    if updated:
        update_all_geometries_on_vis_thread(vis)

    return True

vis.register_animation_callback(animation_callback)

def input_thread_func():
    try:
        import os
        import time
        
        # 检查是否自动模式
        auto_mode = os.environ.get('AUTO_TEXTURE', '').lower() == 'true'
        
        # 尝试加载已保存的旋转配置
        loaded_rotation = None
        if AUTO_LOAD_ROTATION and not auto_mode:
            loaded_rotation = load_rotation_config()
        
        if auto_mode:
            # 自动模式：应用默认旋转并开始
            print("\n🤖 自动模式：应用默认旋转并开始")
            rx, ry, rz = DEFAULT_ROTATION
            # 应用旋转
            for _ in range(int(rx // 90)):
                enqueue_command('1')
                time.sleep(0.3)
            for _ in range(int(ry // 90)):
                enqueue_command('3')
                time.sleep(0.3)
            for _ in range(int(rz // 90)):
                enqueue_command('5')
                time.sleep(0.3)
            time.sleep(0.5)
            enqueue_command('START')
            # 保存配置
            save_rotation_config(rx, ry, rz)
            return
        
        elif loaded_rotation is not None:
            # 自动加载模式：使用已保存的配置
            rx, ry, rz = loaded_rotation
            print(f"\n✅ 使用已保存的旋转配置")
            print(f"   按 [Y] 应用并开始 / [N] 手动调整 / [D] 删除配置")
            response = input("   你的选择: ").strip().upper()
            
            if response == 'Y':
                # 应用已保存的旋转
                print(f"\n🔄 应用旋转: X={rx}°, Y={ry}°, Z={rz}°")
                for _ in range(int(rx // 90)):
                    enqueue_command('1')
                    time.sleep(0.3)
                for _ in range(int(ry // 90)):
                    enqueue_command('3')
                    time.sleep(0.3)
                for _ in range(int(rz // 90)):
                    enqueue_command('5')
                    time.sleep(0.3)
                time.sleep(0.5)
                enqueue_command('START')
                return
            elif response == 'D':
                # 删除配置文件
                try:
                    Path(ROTATION_CONFIG_FILE).unlink()
                    print(f"🗑️  配置文件已删除")
                except:
                    pass
                # 继续手动调整
            else:
                print(f"🎮 进入手动调整模式")
        
        # 手动调整模式
        while not should_quit[0] and not start_texturing[0]:
            print(f"\n[旋转: X={rotation_controller.rx}° Y={rotation_controller.ry}° Z={rotation_controller.rz}°] ", end="")
            print("输入命令 (1-6旋转, L锁定, S保存配置, ENTER开始纹理化, Q退出): ", end="", flush=True)
            cmd = input().strip().upper()
            if not cmd:
                # Enter键：保存配置并开始纹理化
                save_rotation_config(rotation_controller.rx, rotation_controller.ry, rotation_controller.rz)
                enqueue_command('START')
                break
            elif cmd == 'S':
                # 手动保存配置
                save_rotation_config(rotation_controller.rx, rotation_controller.ry, rotation_controller.rz)
                continue
            enqueue_command(cmd)
            if cmd in ('Q', 'QUIT', 'EXIT'):
                break
    except (KeyboardInterrupt, EOFError):
        enqueue_command('Q')

input_thread = threading.Thread(target=input_thread_func, daemon=True)
input_thread.start()

vis.run()
vis.destroy_window()

if should_quit[0]:
    print("\n用户取消，程序退出")
    sys.exit(0)

# ============================================================================
# 第二阶段：实时纹理化
# ============================================================================
print(f"\n" + "="*70)
print(f"阶段 2：实时纹理化")
print(f"="*70)

# 使用确认后的位姿
camera_poses_Twc = world_poses  # 已经是 Twc 格式（world -> cam 的逆）
camera_positions = world_positions

# 保存旋转参数（用于自适应深度检测）
final_rx = rotation_controller.rx
final_ry = rotation_controller.ry
final_rz = rotation_controller.rz

# 加载内参
print(f"\n[1/3] 加载相机内参")
K_data = np.load(K_npz)
K = K_data['K']
if 'W' in K_data and 'H' in K_data:
    W, H = int(K_data['W']), int(K_data['H'])
elif 'width' in K_data and 'height' in K_data:
    W, H = int(K_data['width']), int(K_data['height'])
else:
    raise KeyError("相机内参文件中未找到 'W/H' 或 'width/height' 键")
print(f"   图像尺寸: {W}x{H}")

# 采样
frame_ids = np.arange(len(camera_poses_Twc))
texture_sampled_indices = list(range(0, len(frame_ids), FRAME_SAMPLE_RATE))[:MAX_IMAGES]
sampled_positions = camera_positions[texture_sampled_indices]
sampled_frame_ids = [frame_ids[i] for i in texture_sampled_indices]
sampled_Twc = [camera_poses_Twc[i] for i in texture_sampled_indices]
sampled_Tcw = [np.linalg.inv(T) for T in sampled_Twc]

print(f"   采样: {len(texture_sampled_indices)} 帧")

# 初始化OpenMVS增强功能
print(f"\n[2/3] 初始化OpenMVS增强功能")
vertices_np = np.asarray(mesh.vertices)
normals_np = np.asarray(mesh.vertex_normals)
vertex_colors = np.zeros((len(vertices_np), 3))
vertex_weights = np.zeros(len(vertices_np))
frame_exposures = []
initial_colors = np.full((len(vertices_np), 3), 0.3)
mesh.vertex_colors = o3d.utility.Vector3dVector(initial_colors)

# 上传常用数据到GPU
vertex_colors_gpu = None
vertex_weights_gpu = None
if USE_GPU:
    print(f"\n   📤 上传数据到GPU...")
    vertices_gpu = to_gpu(vertices_np)
    normals_gpu = to_gpu(normals_np)
    K_gpu = to_gpu(K)
    print(f"      ✓ 顶点: {len(vertices_np):,}")
    print(f"      ✓ 法向量: {len(normals_np):,}")
    print(f"      ✓ 相机内参: {K.shape}")
    mem_used, mem_total = check_gpu_memory()
    print(f"      显存使用: {mem_used:.2f} / {mem_total:.1f} GB")
else:
    vertices_gpu = vertices_np
    normals_gpu = normals_np
    K_gpu = K

# 相机可视化辅助函数
def create_camera_frustum(T_wc, scale=0.5):
    frustum_cam = np.array([
        [ 0.0,  0.0,  0.0],
        [-0.5, -0.3,  1.0],
        [ 0.5, -0.3,  1.0],
        [ 0.5,  0.3,  1.0],
        [-0.5,  0.3,  1.0],
    ]) * scale
    frustum_h = np.hstack([frustum_cam, np.ones((frustum_cam.shape[0], 1))])
    frustum_world = (T_wc @ frustum_h.T).T[:, :3]
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(frustum_world)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1,0,0]]*len(lines))
    return ls

def create_camera_coordinate_frame(T_wc, scale=0.3):
    cam_pos = T_wc[:3, 3]
    Rw = T_wc[:3, :3]
    points_list = []
    lines_list = []
    colors_list = []

    def add_axis(dir_cam, color_rgb, n_seg=5):
        nonlocal points_list, lines_list, colors_list
        start_idx = len(points_list)
        for i in range(n_seg):
            t = i / float(n_seg)
            p = cam_pos + Rw @ (dir_cam * (scale * t))
            points_list.append(p)
        points_list.append(cam_pos + Rw @ (dir_cam * scale))
        for i in range(n_seg):
            lines_list.append([start_idx + i, start_idx + i + 1])
            colors_list.append(color_rgb)

    add_axis(np.array([1.0, 0.0, 0.0]), [1, 0, 0])  # X - 红色
    add_axis(np.array([0.0, 1.0, 0.0]), [0, 1, 0])  # Y - 绿色
    add_axis(np.array([0.0, 0.0, 1.0]), [0, 0.5, 1])  # Z - 蓝色

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points_list))
    line_set.lines = o3d.utility.Vector2iVector(lines_list)
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors_list))
    return line_set

# 启动可视化
print(f"\n[3/3] 启动实时可视化")
print(f"\n开始纹理化...")

vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name="实时纹理化 - OpenMVS增强版", width=1600, height=900)
vis2.add_geometry(mesh)

start_sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
start_sphere2.translate(camera_positions[0])
start_sphere2.paint_uniform_color([0, 1, 0])
vis2.add_geometry(start_sphere2)

current_camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
current_camera_sphere.paint_uniform_color([1, 0, 0])
vis2.add_geometry(current_camera_sphere)

current_frustum = o3d.geometry.LineSet()
vis2.add_geometry(current_frustum)

current_frame = o3d.geometry.LineSet()
vis2.add_geometry(current_frame)

processed_trajectory = o3d.geometry.LineSet()
vis2.add_geometry(processed_trajectory)

history_frames = []
for _ in range(20):
    frame = o3d.geometry.LineSet()
    vis2.add_geometry(frame)
    history_frames.append(frame)

render_option2 = vis2.get_render_option()
render_option2.mesh_show_back_face = True
render_option2.light_on = True

# ============================================================================
# 主循环 - OpenMVS增强纹理化
# ============================================================================
print(f"\n{'='*70}")
print(f"开始纹理化处理")
print(f"{'='*70}")
if USE_GPU:
    mem_used, mem_total = check_gpu_memory()
    print(f"🚀 运行模式: GPU加速")
    # 获取GPU名称（兼容不同CuPy版本）
    try:
        device = cp.cuda.Device(GPU_DEVICE_ID)
        gpu_name = device.attributes.get('Name', b'GPU Device')
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        print(f"   GPU设备: {gpu_name}")
    except:
        print(f"   GPU设备: Device {GPU_DEVICE_ID}")
    print(f"   初始显存: {mem_used:.2f} / {mem_total:.1f} GB")
else:
    print(f"💻 运行模式: CPU")
print(f"   总帧数: {len(texture_sampled_indices)}")
print(f"   采样率: 每{FRAME_SAMPLE_RATE}帧")
print(f"{'='*70}")
print(f"\n处理帧...")

frame_count = 0
history_count = 0
start_time = time.time()

# 图像质量统计
quality_stats = {
    'total_frames': 0,
    'skipped_blur': 0,
    'skipped_overexp': 0,
    'skipped_underexp': 0,
    'skipped_quality': 0,
    'processed_frames': 0,
    'avg_quality': [],
    'avg_sharpness': []
}

for i in range(len(texture_sampled_indices)):
    if not vis2.poll_events():
        break

    fid = sampled_frame_ids[i]
    rgb_file = Path(rgb_dir) / f"{fid}.png"
    if not rgb_file.exists():
        rgb_file = Path(rgb_dir) / f"{fid:04d}.png"
    
    depth_file = Path(depth_dir) / f"{fid}.png"
    if not depth_file.exists():
        depth_file = Path(depth_dir) / f"{fid:04d}.png"

    if not rgb_file.exists():
        print(f"\n   警告: RGB文件不存在: {rgb_file}")
        continue

    rgb_img = cv2.imread(str(rgb_file))
    if rgb_img is None:
        continue
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) / 255.0

    # ========================================================================
    # 图像预处理增强（第2步优化）- 在质量评估之前应用
    # ========================================================================
    if USE_IMAGE_ENHANCEMENT:
        rgb_img = enhance_image(rgb_img)

    # ========================================================================
    # 图像质量评估与过滤（第1步优化）
    # ========================================================================
    if USE_IMAGE_QUALITY_FILTER:
        quality_stats['total_frames'] += 1
        quality_score, sharpness, overexposed, underexposed, contrast = assess_image_quality(rgb_img)
        
        # 检查是否满足质量标准
        quality_passed = True
        skip_reasons = []
        
        if quality_score < IMAGE_QUALITY_THRESHOLD:
            quality_passed = False
            skip_reasons.append(f"质量分={quality_score:.1f}<{IMAGE_QUALITY_THRESHOLD:.1f}")
            quality_stats['skipped_quality'] += 1
        
        if sharpness < SHARPNESS_THRESHOLD:
            quality_passed = False
            skip_reasons.append(f"模糊(清晰度={sharpness:.1f}<{SHARPNESS_THRESHOLD:.1f})")
            quality_stats['skipped_blur'] += 1
        
        if overexposed > MAX_OVEREXPOSURE:
            quality_passed = False
            skip_reasons.append(f"过曝({overexposed*100:.1f}%>{MAX_OVEREXPOSURE*100:.1f}%)")
            quality_stats['skipped_overexp'] += 1
        
        if underexposed > MAX_UNDEREXPOSURE:
            quality_passed = False
            skip_reasons.append(f"欠曝({underexposed*100:.1f}%>{MAX_UNDEREXPOSURE*100:.1f}%)")
            quality_stats['skipped_underexp'] += 1
        
        # 如果质量不合格，跳过此帧
        if not quality_passed:
            if SHOW_QUALITY_STATS:
                reasons_str = ", ".join(skip_reasons)
                print(f"   [跳过帧{fid:04d}] {reasons_str}")
            continue
        
        # 记录通过的帧的质量指标
        quality_stats['processed_frames'] += 1
        quality_stats['avg_quality'].append(quality_score)
        quality_stats['avg_sharpness'].append(sharpness)
        
        # 显示质量统计（仅对通过的帧）
        if SHOW_QUALITY_STATS and frame_count % 50 == 0:  # 每50帧显示一次
            print(f"   [帧{fid:04d}] 质量={quality_score:.1f}, 清晰度={sharpness:.1f}, "
                  f"过曝={overexposed*100:.1f}%, 欠曝={underexposed*100:.1f}%, 对比度={contrast:.1f}")

    depth_img = None
    if USE_DEPTH_CONSISTENCY and depth_file.exists():
        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            depth_img = depth_img.astype(np.float32) / DEPTH_SCALE

    if USE_EXPOSURE_COMP:
        frame_brightness = np.mean(rgb_img)
        frame_exposures.append(frame_brightness)
        if len(frame_exposures) > 1:
            avg_brightness = np.mean(frame_exposures)
            exposure_factor = avg_brightness / (frame_brightness + 1e-6)
            rgb_img = np.clip(rgb_img * exposure_factor, 0, 1)

    T_wc_use = sampled_Twc[i]
    T_cw_use = sampled_Tcw[i]
    cam_pos = T_wc_use[:3, 3]

    # 更新可视化
    sphere_center = np.asarray(current_camera_sphere.get_center())
    current_camera_sphere.translate(cam_pos - sphere_center)
    vis2.update_geometry(current_camera_sphere)

    if SHOW_CAMERA_FRUSTUM:
        new_frustum = create_camera_frustum(T_wc_use, scale=2.0)
        current_frustum.points = new_frustum.points
        current_frustum.lines = new_frustum.lines
        current_frustum.colors = new_frustum.colors
        vis2.update_geometry(current_frustum)

    new_frame = create_camera_coordinate_frame(T_wc_use, scale=1.5)
    current_frame.points = new_frame.points
    current_frame.lines = new_frame.lines
    current_frame.colors = new_frame.colors
    vis2.update_geometry(current_frame)

    if i % 5 == 0 and history_count < len(history_frames):
        history_frame = create_camera_coordinate_frame(T_wc_use, scale=0.8)
        history_frames[history_count].points = history_frame.points
        history_frames[history_count].lines = history_frame.lines
        history_colors = np.array(history_frame.colors) * 0.5
        history_frames[history_count].colors = o3d.utility.Vector3dVector(history_colors)
        vis2.update_geometry(history_frames[history_count])
        history_count += 1

    if i > 0:
        trajectory_points = sampled_positions[:i+1]
        trajectory_lines = [[j, j+1] for j in range(len(trajectory_points)-1)]
        processed_trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
        processed_trajectory.lines = o3d.utility.Vector2iVector(trajectory_lines)
        processed_trajectory.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in trajectory_lines])
        vis2.update_geometry(processed_trajectory)

    # OpenMVS增强纹理映射
    if USE_GPU:
        # GPU加速版本 - 更多操作在GPU上完成
        xp = cp
        T_cw_gpu = to_gpu(T_cw_use)
        
        # 顶点变换到相机坐标系（GPU）
        ones_gpu = xp.ones((len(vertices_gpu), 1), dtype=xp.float32)
        vertices_hom_gpu = xp.hstack([vertices_gpu, ones_gpu])
        vertices_cam_gpu = (T_cw_gpu @ vertices_hom_gpu.T).T
        visible_mask_gpu = vertices_cam_gpu[:, 2] > MIN_DEPTH
        
        if xp.any(visible_mask_gpu):
            # 投影计算（GPU）
            vertices_cam_vis_gpu = vertices_cam_gpu[visible_mask_gpu, :3]
            
            # 第7步优化：使用亚像素精度投影（GPU版本）
            if USE_SUBPIXEL_PRECISION:
                u_f_gpu, v_f_gpu = project_vertices_subpixel_gpu(
                    vertices_cam_vis_gpu, K_gpu, H, W, use_float64=USE_FLOAT64_PROJECTION
                )
            else:
                # 传统投影（float32）
                points_2d_gpu = K_gpu @ vertices_cam_vis_gpu.T
                u_f_gpu = points_2d_gpu[0, :] / points_2d_gpu[2, :]
                v_f_gpu = points_2d_gpu[1, :] / points_2d_gpu[2, :]
                v_f_gpu = H - 1 - v_f_gpu
            
            # 边界检查（GPU）
            in_bounds_gpu = (u_f_gpu >= 0.5) & (u_f_gpu < W-0.5) & (v_f_gpu >= 0.5) & (v_f_gpu < H-0.5)
            
            # 只传输必要的数据到CPU
            if xp.any(in_bounds_gpu):
                visible_mask = to_cpu(visible_mask_gpu)
                in_bounds = to_cpu(in_bounds_gpu)
                u_f = to_cpu(u_f_gpu[in_bounds_gpu])
                v_f = to_cpu(v_f_gpu[in_bounds_gpu])
                vertices_cam_vis = to_cpu(vertices_cam_vis_gpu)
            else:
                visible_mask = to_cpu(visible_mask_gpu)
                in_bounds = np.array([])
                u_f = np.array([])
                v_f = np.array([])
                vertices_cam_vis = to_cpu(vertices_cam_vis_gpu)
        else:
            visible_mask = to_cpu(visible_mask_gpu)
            in_bounds = np.array([])
            u_f = np.array([])
            v_f = np.array([])
            vertices_cam_vis = np.array([])
    else:
        # CPU版本
        vertices_hom = np.hstack([vertices_np, np.ones((len(vertices_np), 1))])
        vertices_cam = (T_cw_use @ vertices_hom.T).T
        visible_mask = vertices_cam[:, 2] > MIN_DEPTH
        
        if np.any(visible_mask):
            vertices_cam_vis = vertices_cam[visible_mask, :3]
            
            # 第7步优化：使用亚像素精度投影
            if USE_SUBPIXEL_PRECISION:
                u_f, v_f = project_vertices_subpixel(
                    vertices_cam_vis, K, H, W, use_float64=USE_FLOAT64_PROJECTION
                )
            else:
                # 传统投影（float32）
                points_2d = K @ vertices_cam_vis.T
                u_f = points_2d[0, :] / points_2d[2, :]
                v_f = points_2d[1, :] / points_2d[2, :]
                v_f = H - 1 - v_f
            
            in_bounds = (u_f >= 0.5) & (u_f < W-0.5) & (v_f >= 0.5) & (v_f < H-0.5)
        else:
            in_bounds = np.array([])
            u_f = np.array([])
            v_f = np.array([])
            vertices_cam_vis = np.array([])

    if np.any(visible_mask) and np.any(in_bounds):
            if not USE_GPU:
                u_f = u_f[in_bounds]
                v_f = v_f[in_bounds]
                vertices_cam_in = vertices_cam_vis[in_bounds]
            else:
                # GPU版本已经过滤
                vertices_cam_in = vertices_cam_vis[in_bounds]

            # 深度一致性检测
            depth_consistency_mask = np.ones(len(u_f), dtype=bool)
            if USE_DEPTH_CONSISTENCY and depth_img is not None:
                u_int = np.clip(u_f.astype(int), 0, W-1)
                v_int = np.clip(v_f.astype(int), 0, H-1)
                depth_measured = depth_img[v_int, u_int]
                depth_rendered = vertices_cam_in[:, 2]
                
                # 根据配置选择深度检测方法
                if USE_ADAPTIVE_DEPTH:
                    # 自适应深度检测：根据表面类型（地面/墙壁）使用不同阈值
                    # 需要获取世界坐标系下的法向量
                    valid_indices_depth = np.where(visible_mask)[0][in_bounds]
                    vertex_normals_world = normals_np[valid_indices_depth]
                    
                    # 计算向上方向（考虑旋转）
                    R_world = R.from_euler('xyz', [final_rx, final_ry, final_rz], degrees=True).as_matrix()
                    up_vector = R_world @ np.array([0, 0, 1])  # 初始Z轴旋转到世界坐标
                    
                    depth_consistency_mask = adaptive_depth_test(
                        depth_rendered, depth_measured, vertex_normals_world,
                        up_vector=up_vector,
                        floor_normal_threshold=FLOOR_NORMAL_THRESHOLD,
                        floor_depth_front=FLOOR_DEPTH_FRONT,
                        floor_depth_back=FLOOR_DEPTH_BACK,
                        wall_depth_front=WALL_DEPTH_FRONT,
                        wall_depth_back=WALL_DEPTH_BACK,
                        min_depth=MIN_DEPTH,
                        max_depth=MAX_DEPTH
                    )
                else:
                    # 传统深度检测：使用单一阈值
                    depth_diff = np.abs(depth_rendered - depth_measured)
                    depth_consistency_mask = (depth_diff < DEPTH_THRESHOLD) & \
                                             (depth_measured > MIN_DEPTH) & \
                                             (depth_measured < MAX_DEPTH)

            if not np.any(depth_consistency_mask):
                vis2.update_renderer()
                frame_count += 1
                continue

            u_f = u_f[depth_consistency_mask]
            v_f = v_f[depth_consistency_mask]
            vertices_cam_in = vertices_cam_in[depth_consistency_mask]
            
            # 车辆检测与去除（四模态几何检测）
            vehicle_removal_mask = np.ones(len(u_f), dtype=bool)
            if USE_VEHICLE_DETECTION and depth_img is not None:
                # 创建完整的渲染深度图（TSDF理论深度）
                depth_rendered_img = np.zeros((H, W), dtype=np.float32)
                u_int_temp = np.clip(u_f.astype(int), 0, W-1)
                v_int_temp = np.clip(v_f.astype(int), 0, H-1)
                depth_rendered_temp = vertices_cam_in[:, 2]
                depth_rendered_img[v_int_temp, u_int_temp] = depth_rendered_temp
                
                # 准备法向量和3D点（如果需要）
                normals_img = None
                points_3d_img = None
                
                if USE_GROUND_NORMAL or USE_HEIGHT_FILTER:
                    # 创建法向量和3D点的图像
                    normals_img = np.zeros((H, W, 3), dtype=np.float32)
                    points_3d_img = np.zeros((H, W, 3), dtype=np.float32)
                    
                    # 从相机坐标系转换到世界坐标系
                    vertices_world = (T_wc_use[:3, :3] @ vertices_cam_in.T + T_wc_use[:3, 3:4]).T
                    normals_world = T_wc_use[:3, :3] @ normals_np[visible_mask][in_bounds][depth_consistency_mask].T
                    normals_world = normals_world.T
                    
                    # 填充图像
                    normals_img[v_int_temp, u_int_temp] = normals_world
                    points_3d_img[v_int_temp, u_int_temp] = vertices_world
                
                # 四模态车辆检测
                vehicle_mask_full, cue_masks = detect_vehicles_multimodal(
                    depth_obs=depth_img,
                    depth_tsdf=depth_rendered_img,
                    normals=normals_img,
                    points_3d=points_3d_img,
                    config=None  # 使用全局配置
                )
                
                # 细化mask（形态学处理）
                vehicle_mask_full = refine_vehicle_mask(
                    vehicle_mask_full, 
                    dilation=VEHICLE_MASK_DILATION
                )
                
                # 可选：保存vehicle mask用于调试
                if SAVE_VEHICLE_MASKS and frame_count % 50 == 0:
                    mask_save_dir = Path("output/vehicle_masks")
                    mask_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存综合掩码
                    mask_vis = (vehicle_mask_full * 255).astype(np.uint8)
                    cv2.imwrite(str(mask_save_dir / f"mask_{frame_count:04d}.png"), mask_vis)
                    
                    # 保存各线索的独立结果
                    for cue_name, cue_mask in cue_masks.items():
                        if cue_mask is not None:
                            cue_vis = (cue_mask * 255).astype(np.uint8)
                            cv2.imwrite(str(mask_save_dir / f"cue_{cue_name}_{frame_count:04d}.png"), cue_vis)
                
                # 过滤掉车辆像素
                vehicle_pixel_mask = vehicle_mask_full[v_int_temp, u_int_temp]
                vehicle_removal_mask = ~vehicle_pixel_mask
                
                # 统计信息
                n_vehicle_pixels = np.sum(vehicle_pixel_mask)
                if DEBUG_MODE and n_vehicle_pixels > 0:
                    n_total = len(vehicle_pixel_mask)
                    percentage = (n_vehicle_pixels / n_total) * 100
                    print(f"      🚗 检测车辆像素: {n_vehicle_pixels}/{n_total} ({percentage:.1f}%)")
                    
                    # 显示各线索贡献
                    if frame_count % 100 == 0:
                        for cue_name, cue_mask in cue_masks.items():
                            if cue_mask is not None:
                                cue_count = np.sum(cue_mask[v_int_temp, u_int_temp])
                                print(f"         - {cue_name}: {cue_count}/{n_total} ({cue_count/n_total*100:.1f}%)")
            
            if not np.any(vehicle_removal_mask):
                vis2.update_renderer()
                frame_count += 1
                continue
            
            u_f = u_f[vehicle_removal_mask]
            v_f = v_f[vehicle_removal_mask]
            vertices_cam_in = vertices_cam_in[vehicle_removal_mask]

            # 第4步优化：智能视角选择与加权
            valid_indices_temp = np.where(visible_mask)[0][in_bounds][depth_consistency_mask][vehicle_removal_mask]
            
            if USE_SMART_VIEW_WEIGHTING:
                # 计算视角质量权重
                vertex_normals = normals_np[valid_indices_temp]
                view_dirs = -vertices_cam_in / np.linalg.norm(vertices_cam_in, axis=1, keepdims=True)
                R_cw_use = T_cw_use[:3, :3]
                normals_cam = (R_cw_use @ vertex_normals.T).T
                normals_cam = normals_cam / np.linalg.norm(normals_cam, axis=1, keepdims=True)
                
                # 视角权重（基于法向-视线夹角）
                view_weights = compute_view_angle_weight(
                    normals_cam, view_dirs, max_angle_deg=MAX_VIEW_ANGLE_DEG
                )
                
                # 距离权重（基于相机距离）
                distances = np.linalg.norm(vertices_cam_in, axis=1)
                dist_weights = compute_distance_weight(distances, falloff=DISTANCE_FALLOFF)
                
                # 图像质量权重（归一化到[0,1]）
                if USE_IMAGE_QUALITY_FILTER and quality_score > 0:
                    # 使用当前帧的质量分数（所有像素相同）
                    quality_weights = np.full(len(u_f), quality_score / 100.0)
                else:
                    quality_weights = np.ones(len(u_f))
                
                # 综合权重
                combined_weights = compute_combined_weight(
                    view_weights, dist_weights, quality_weights,
                    view_alpha=VIEW_ANGLE_WEIGHT,
                    dist_alpha=DISTANCE_WEIGHT,
                    quality_alpha=IMAGE_QUALITY_WEIGHT
                )
                
                # 过滤低权重采样
                effective_mask = combined_weights >= MIN_EFFECTIVE_WEIGHT
                
                if not np.any(effective_mask):
                    vis2.update_renderer()
                    frame_count += 1
                    continue
                
                u_f = u_f[effective_mask]
                v_f = v_f[effective_mask]
                valid_indices_temp = valid_indices_temp[effective_mask]
                angle_weights = combined_weights[effective_mask]
                
            elif USE_ANGLE_WEIGHTING:
                # 传统视角加权（向后兼容）
                vertex_normals = normals_np[valid_indices_temp]
                view_dirs = -vertices_cam_in / np.linalg.norm(vertices_cam_in, axis=1, keepdims=True)
                R_cw_use = T_cw_use[:3, :3]
                normals_cam = (R_cw_use @ vertex_normals.T).T
                normals_cam = normals_cam / np.linalg.norm(normals_cam, axis=1, keepdims=True)
                cos_angles = np.sum(normals_cam * view_dirs, axis=1)
                cos_angles = np.clip(cos_angles, -1, 1)
                angles_deg = np.degrees(np.arccos(cos_angles))
                angle_mask = angles_deg < ANGLE_THRESHOLD_DEG

                if not np.any(angle_mask):
                    vis2.update_renderer()
                    frame_count += 1
                    continue

                u_f = u_f[angle_mask]
                v_f = v_f[angle_mask]
                cos_angles = cos_angles[angle_mask]
                valid_indices_temp = valid_indices_temp[angle_mask]
                angle_weights = cos_angles ** 2
            else:
                # 无加权
                angle_weights = np.ones(len(u_f))

            # 纹理采样（双三次插值 或 双线性插值）
            if USE_SUBPIXEL_PRECISION and PRESERVE_SUBPIXEL_WEIGHT:
                # 第7步优化：使用亚像素精度采样（保留小数精度）
                subpixel_weights = compute_subpixel_weights(u_f, v_f, mode=SUBPIXEL_WEIGHT_MODE)
                colors = sample_with_subpixel_weights(rgb_img, subpixel_weights, H, W)
            elif USE_BICUBIC_INTERPOLATION:
                # 第3步优化：使用双三次插值（更高质量）
                colors = bicubic_interpolate(rgb_img, u_f, v_f, a=BICUBIC_A)
            else:
                # 传统双线性插值
                u0 = np.floor(u_f).astype(int)
                v0 = np.floor(v_f).astype(int)
                u1 = np.minimum(u0 + 1, W - 1)
                v1 = np.minimum(v0 + 1, H - 1)
                wu = u_f - u0
                wv = v_f - v0
                c00 = rgb_img[v0, u0]
                c10 = rgb_img[v0, u1]
                c01 = rgb_img[v1, u0]
                c11 = rgb_img[v1, u1]
                colors = (c00 * (1-wu)[:, np.newaxis] * (1-wv)[:, np.newaxis] +
                          c10 *  wu[:, np.newaxis] * (1-wv)[:, np.newaxis] +
                          c01 * (1-wu)[:, np.newaxis] *  wv[:, np.newaxis] +
                          c11 *  wu[:, np.newaxis] *  wv[:, np.newaxis])

            # 加权融合
            valid_indices = valid_indices_temp
            vertex_colors[valid_indices] += colors * angle_weights[:, np.newaxis]
            vertex_weights[valid_indices] += angle_weights

            # 更新颜色（降低更新频率以提高GPU利用率）
            if frame_count % VISUALIZATION_UPDATE_RATE == 0:
                current_colors = initial_colors.copy()
                non_zero = vertex_weights > 0
                current_colors[non_zero] = vertex_colors[non_zero] / vertex_weights[non_zero, np.newaxis]
                mesh.vertex_colors = o3d.utility.Vector3dVector(current_colors)
                vis2.update_geometry(mesh)

    vis2.update_renderer()
    frame_count += 1
    progress = frame_count / len(texture_sampled_indices) * 100
    colored_vertices = np.sum(vertex_weights > 0)
    coverage = colored_vertices / len(vertices_np) * 100

    if DEBUG_MODE and frame_count <= 5:
        cam_x = T_wc_use[:3, 0]
        cam_y = T_wc_use[:3, 1]
        cam_z = T_wc_use[:3, 2]
        print(f"\n  [调试 帧{fid}]")
        print(f"    - 相机位置: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
        print(f"    - X轴(红): [{cam_x[0]:.3f}, {cam_x[1]:.3f}, {cam_x[2]:.3f}]")
        print(f"    - Y轴(绿): [{cam_y[0]:.3f}, {cam_y[1]:.3f}, {cam_y[2]:.3f}]")
        print(f"    - Z轴(蓝): [{cam_z[0]:.3f}, {cam_z[1]:.3f}, {cam_z[2]:.3f}]")
        if USE_GPU:
            mem_used, mem_total = check_gpu_memory()
            print(f"    - GPU显存: {mem_used:.2f} / {mem_total:.1f} GB")

    # 进度显示
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    eta = (len(texture_sampled_indices) - frame_count) / fps if fps > 0 else 0
    
    progress_str = f"\r  [{frame_count}/{len(texture_sampled_indices)}] {progress:.1f}% | " \
                   f"覆盖: {coverage:.1f}% | " \
                   f"帧{fid:04d} | "
    
    if USE_GPU:
        mem_used, mem_total = check_gpu_memory()
        mem_usage_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
        progress_str += f"🚀GPU: {mem_used:.1f}/{mem_total:.1f}GB ({mem_usage_pct:.0f}%) | "
    else:
        progress_str += f"💻CPU | "
    
    progress_str += f"{fps:.1f}fps | ETA: {eta:.0f}s"
    print(progress_str, end="", flush=True)

    time.sleep(UPDATE_INTERVAL)

print("\n\n" + "="*70)
print("[SUCCESS] OpenMVS增强纹理化完成!")
print("="*70)

# 从GPU下载最终结果
if USE_GPU:
    print(f"\n📥 释放GPU资源...")
    # 清理GPU内存
    del vertices_gpu
    del normals_gpu
    del K_gpu
    if vertex_colors_gpu is not None:
        del vertex_colors_gpu
    if vertex_weights_gpu is not None:
        del vertex_weights_gpu
    if cp is not None:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    print(f"   ✓ GPU内存已释放")

# ============================================================================
# 第5步优化：接缝平滑后处理
# ============================================================================
n_seam_vertices = 0  # 初始化接缝顶点数
seam_ratio = 0.0

if USE_SEAM_SMOOTHING:
    print(f"\n🎨 应用接缝平滑（第5步优化）...")
    seam_start_time = time.time()
    
    # 检测接缝区域
    print(f"   [1/3] 检测接缝区域...")
    seam_mask, variances = detect_seam_regions(
        mesh, vertex_colors, vertex_weights,
        variance_threshold=VARIANCE_THRESHOLD,
        k_neighbors=SEAM_K_NEIGHBORS
    )
    
    n_seam_vertices = np.sum(seam_mask)
    seam_ratio = n_seam_vertices / len(vertices_np) * 100
    print(f"         ✓ 检测到 {n_seam_vertices:,} 个接缝顶点 ({seam_ratio:.1f}%)")
    
    if n_seam_vertices > 0:
        # 应用平滑
        print(f"   [2/3] 应用自适应平滑...")
        smoothed_colors = apply_adaptive_smoothing(
            mesh, vertex_colors, vertex_weights, seam_mask,
            smoothing_strength=SMOOTHING_STRENGTH,
            k_neighbors=SEAM_K_NEIGHBORS
        )
        
        # 更新顶点颜色
        print(f"   [3/3] 更新网格颜色...")
        mesh.vertex_colors = o3d.utility.Vector3dVector(smoothed_colors)
        
        seam_time = time.time() - seam_start_time
        print(f"   ✓ 接缝平滑完成! 耗时: {seam_time:.2f}秒")
    else:
        print(f"   ℹ 未检测到明显接缝，跳过平滑")
else:
    # 不使用接缝平滑，直接归一化顶点颜色
    final_colors = initial_colors.copy()
    non_zero = vertex_weights > 0
    final_colors[non_zero] = vertex_colors[non_zero] / vertex_weights[non_zero, np.newaxis]
    mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

# ============================================================================
# 后处理：填充空白区域（车辆遮挡导致的未投影区域）
# ============================================================================
if FILL_EMPTY_VERTICES and USE_VEHICLE_REMOVAL:
    print(f"\n🎨 填充空白区域（后处理）...")
    fill_start_time = time.time()
    
    # 获取当前颜色和权重
    current_colors = np.asarray(mesh.vertex_colors)
    
    # 使用K近邻填充
    if FILL_METHOD == 'knn':
        filled_colors = fill_empty_vertices(
            current_colors, 
            vertex_weights, 
            vertices_np,
            k=KNN_NEIGHBORS
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(filled_colors)
    
    fill_time = time.time() - fill_start_time
    print(f"   ✓ 填充完成! 耗时: {fill_time:.2f}秒")
elif FILL_EMPTY_VERTICES and not USE_VEHICLE_REMOVAL:
    print(f"\n⚠️  填充空白区域需要启用车辆去除功能")

colored_vertices = np.sum(vertex_weights > 0)
coverage = colored_vertices / len(vertices_np) * 100
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\n最终统计:")
print(f"  - 计算设备: {'🚀 GPU加速' if USE_GPU else '💻 CPU'}")
print(f"  - 处理帧数: {frame_count}")

# 显示图像质量统计
if USE_IMAGE_QUALITY_FILTER and quality_stats['total_frames'] > 0:
    print(f"\n图像质量统计（第1步优化）:")
    print(f"  - 检查帧数: {quality_stats['total_frames']}")
    print(f"  - 通过帧数: {quality_stats['processed_frames']} ({quality_stats['processed_frames']/quality_stats['total_frames']*100:.1f}%)")
    print(f"  - 跳过帧数: {quality_stats['total_frames'] - quality_stats['processed_frames']} ({(quality_stats['total_frames'] - quality_stats['processed_frames'])/quality_stats['total_frames']*100:.1f}%)")
    print(f"    └─ 模糊: {quality_stats['skipped_blur']}")
    print(f"    └─ 过曝: {quality_stats['skipped_overexp']}")
    print(f"    └─ 欠曝: {quality_stats['skipped_underexp']}")
    print(f"    └─ 质量低: {quality_stats['skipped_quality']}")
    if quality_stats['avg_quality']:
        print(f"  - 平均质量分数: {np.mean(quality_stats['avg_quality']):.1f}")
        print(f"  - 平均清晰度: {np.mean(quality_stats['avg_sharpness']):.1f}")
print(f"  - 总耗时: {total_time:.1f}秒")
print(f"  - 平均速度: {avg_fps:.2f} fps")
print(f"  - 每帧耗时: {total_time/frame_count*1000:.1f} ms") if frame_count > 0 else None
print(f"  - 着色顶点: {colored_vertices:,}")
print(f"  - 覆盖率: {coverage:.1f}%")
print(f"  - 深度一致性: {'已启用' if USE_DEPTH_CONSISTENCY else '未启用'}")
if USE_SMART_VIEW_WEIGHTING:
    print(f"  - 智能视角加权: 已启用 (第4步优化)")
    print(f"    └─ 权重: 视角{VIEW_ANGLE_WEIGHT:.0%} + 距离{DISTANCE_WEIGHT:.0%} + 质量{IMAGE_QUALITY_WEIGHT:.0%}")
elif USE_ANGLE_WEIGHTING:
    print(f"  - 传统视角加权: 已启用")
else:
    print(f"  - 视角加权: 未启用")
if USE_SEAM_SMOOTHING and n_seam_vertices > 0:
    print(f"  - 接缝平滑: 已启用 (第5步优化)")
    print(f"    └─ 平滑顶点: {n_seam_vertices:,} ({seam_ratio:.1f}%), 强度: {SMOOTHING_STRENGTH:.0%}")
elif USE_SEAM_SMOOTHING:
    print(f"  - 接缝平滑: 已启用 (未检测到接缝)")
else:
    print(f"  - 接缝平滑: 未启用")
print(f"  - 曝光补偿: {'已启用' if USE_EXPOSURE_COMP else '未启用'}")
print(f"  - 纹理插值: {'双三次 (更高质量)' if USE_BICUBIC_INTERPOLATION else '双线性'}")

# 第6步优化结束统计
if USE_LAB_COLOR_SPACE:
    print(f"  - LAB色彩空间: 已启用 (第6步优化)")
    print(f"    └─ 用于接缝平滑的颜色混合")
else:
    print(f"  - LAB色彩空间: 未启用 (使用RGB空间)")

if USE_GPU:
    print(f"\n💡 提示: GPU加速模式可提供3-6倍性能提升")
print(f"\n窗口保持打开，按 Q 或 ESC 关闭")

vis2.run()
vis2.destroy_window()

# ============================================================================
# 保存纹理化后的模型
# ============================================================================
if SAVE_TEXTURED_MESH:
    print("\n" + "="*70)
    print("保存纹理化模型")
    print("="*70)
    
    # 创建输出目录
    from pathlib import Path
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    if AUTO_TIMESTAMP:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 分离文件名和扩展名
        filename_base, ext = OUTPUT_FILENAME.rsplit('.', 1) if '.' in OUTPUT_FILENAME else (OUTPUT_FILENAME, 'ply')
        output_filename = f"{filename_base}_{timestamp}.{ext}"
    else:
        output_filename = OUTPUT_FILENAME
    
    output_path = output_dir / output_filename
    
    # 保存模型
    print(f"\n💾 保存模型到: {output_path}")
    try:
        success = o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)
        if success:
            print(f"✅ 模型保存成功!")
            
            # 显示统计信息
            n_vertices = len(np.asarray(mesh.vertices))
            n_triangles = len(np.asarray(mesh.triangles))
            has_colors = mesh.has_vertex_colors()
            print(f"\n模型统计:")
            print(f"  - 顶点数: {n_vertices:,}")
            print(f"  - 三角形数: {n_triangles:,}")
            print(f"  - 顶点颜色: {'是' if has_colors else '否'}")
            
            # 文件大小
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  - 文件大小: {file_size:.2f} MB")
        else:
            print(f"❌ 模型保存失败!")
    except Exception as e:
        print(f"❌ 保存模型时出错: {e}")

print("\n" + "="*70)



