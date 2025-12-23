# Datasets Directory

This directory is for storing your dataset files.

## 📥 Download Datasets

### Dataset Download Links

**1. Complete Dataset (RGB, Depth, Pose, First Frame Pose, Camera Intrinsics)**

- **Quark Cloud Drive**: https://pan.quark.cn/s/ab5c6a9fa61d
- **Extraction Code**: `VC1Y`

Includes:
- RGB color images (color/)
- Depth maps (depth/)
- Camera poses (pose/)
- First frame pose (first_pose_osm.txt)
- Camera intrinsics (K_rectified.npz)

---

**2. Original TSDF File Generated from OSM**

- **Quark Cloud Drive**: https://pan.quark.cn/s/d9c2be04b0f8
- **Extraction Code**: `mLfv`

TSDF (Truncated Signed Distance Function) mesh model generated from OpenStreetMap data.

---

**3. Coarse-Grained TSDF Model (Quick Verification)**

- **Quark Cloud Drive**: https://pan.quark.cn/s/d9c2be04b0f8
- **Extraction Code**: `mLfv`

> ⚠️ **Note**: This model has coarser granularity and smaller file size for quick verification, but may result in reduced clarity.

---

**4. TSDF Split into 4 Blocks**

- **Quark Cloud Drive**: https://pan.quark.cn/s/23a4c2e91556
- **Extraction Code**: `6zYP`

The TSDF model is split into 4 blocks. Users can load blocks separately as needed and merge them into a complete model.

---

**5. Merge Code**

- **Quark Cloud Drive**: https://pan.quark.cn/s/e1abf39d61a8
- **Extraction Code**: `sj5f`

Python script to merge split TSDF blocks into a complete model.

### Download Instructions

1. **Click the Quark Cloud Drive link** above
2. **Enter the extraction code**
3. **Download the file** to your local machine
4. **Extract to `Datasets/` directory**:
   ```bash
   # Example: Extract complete dataset
   cd Datasets/
   unzip dataset.zip
   # or
   tar -xzf dataset.tar.gz
   ```
5. **Verify the structure** matches the format below
6. **Update the config** file (`configs/config_default.yaml`) with the correct paths

### Using Split TSDF Models

If you downloaded the split TSDF model (option 4), use the merge code (option 5) to combine them:

```bash
# Download and extract merge code
# Run merge script
python merge_tsdf_blocks.py \
  --input_dir Datasets/tsdf_blocks/ \
  --output Datasets/your_dataset/mesh.ply
```

**Alternative**: You can use your own dataset by following the data format requirements below.

### Quick Start with Downloaded Data

After downloading and extracting the dataset:

```bash
# Update config file
# Edit configs/config_default.yaml:
#   mesh_path: "Datasets/your_dataset/mesh.ply"
#   pose_dir: "Datasets/your_dataset/pose"
#   rgb_dir: "Datasets/your_dataset/color"
#   depth_dir: "Datasets/your_dataset/depth"
#   K_npz: "Datasets/your_dataset/K_rectified.npz"
#   first_pose_txt: "Datasets/your_dataset/first_pose_osm.txt"

# Run texture mapping
python scripts/texture_realtime.py --config configs/config_default.yaml
```

---

## Directory Structure

Organize your dataset as follows:

```
Datasets/
├── your_dataset/              # Your dataset folder name
│   ├── pose/                  # Camera poses directory
│   │   ├── 0000.txt          # 4x4 transformation matrix (T_wc)
│   │   ├── 0001.txt
│   │   ├── 0002.txt
│   │   └── ...
│   ├── first_pose_osm.txt    # First frame pose (4x4 matrix)
│   ├── color/                # RGB images directory
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── depth/                # Depth images directory
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── K_rectified.npz       # Camera intrinsics
│   └── mesh.ply              # 3D mesh (from TSDF download)
└── ICPARKOSM/
    └── ICPARK.osm            # Example OSM file (for OSM to TSDF conversion)
```

## Data Format Requirements

### Pose Files (`pose/*.txt`)

Each pose file should contain a 4×4 transformation matrix representing the camera pose in world coordinates (T_wc):

```
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0   0   0   1
```

Where:
- `r11` to `r33`: Rotation matrix (3×3)
- `tx`, `ty`, `tz`: Translation vector
- Last row: Always `0 0 0 1`

### First Frame Pose (`first_pose.txt`)

Same format as pose files above. This represents the absolute pose of the first frame in world coordinates.

**Note**: You can use `scripts/osm_pose_selector.py` to interactively select this pose on an OSM map.

### Camera Intrinsics (`K_rectified.npz`)

NPZ file containing:
- `K`: 3×3 camera intrinsic matrix
  ```
  [fx  0  cx]
  [0  fy  cy]
  [0   0   1]
  ```
- `W`, `H` or `width`, `height`: Image dimensions (width and height in pixels)

**Example Python code to create intrinsics file:**
```python
import numpy as np

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

np.savez('K_rectified.npz', K=K, W=width, H=height)
```

### RGB Images (`color/*.png`)

- Format: PNG
- Bit depth: 8-bit or 16-bit
- Color space: RGB

### Depth Images (`depth/*.png`)

- Format: PNG
- Bit depth: 16-bit unsigned integer
- Units: Depth in millimeters (will be divided by `depth_scale` in config, default 1000.0)
- Invalid pixels: Should be 0

### Mesh File (`mesh.ply`)

- Format: PLY (ASCII or binary)
- Required: Vertex positions, triangle faces
- Optional: Vertex normals (will be computed if missing)

**Note**: If you don't have a mesh, you can generate one from OSM data using `scripts/Osm2Tsdf.py`.

## Example OSM File

The `ICPARKOSM/ICPARK.osm` file is provided as an example for testing OSM to TSDF conversion:

```bash
python scripts/Osm2Tsdf.py \
  --osm Datasets/ICPARKOSM/ICPARK.osm \
  --outdir output/ICPARKOSM_generated \
  --voxel 0.10 \
  --height 3.0 \
  --trunc 0.40
```

## Notes

- **File naming**: Should be sequential integers with 4 digits (0000.txt, 0001.txt, ... or 0000.png, 0001.png, ...)
- **Frame matching**: The number of pose files should match the number of RGB/depth images
- **Path configuration**: All paths in `configs/config_default.yaml` should be relative to the project root
- **Git exclusion**: Large data files are excluded from git (see `.gitignore`)

## Preparing Your Own Dataset

### From SLAM/MVS Systems

If you have data from SLAM or MVS systems, convert it to the required format:

1. **Extract poses**: Export camera poses as 4×4 transformation matrices (T_wc format)
2. **Export images**: RGB images as PNG (8-bit)
3. **Export depth**: Depth maps as PNG (16-bit, values in millimeters)
4. **Camera calibration**: Extract intrinsic parameters (fx, fy, cx, cy)
5. **Create mesh**: Either use existing mesh or generate from OSM data

### File Naming Convention

**Important**: All files should use 4-digit naming:
- Pose files: `0000.txt`, `0001.txt`, `0002.txt`, ...
- Color images: `0000.png`, `0001.png`, `0002.png`, ...
- Depth images: `0000.png`, `0001.png`, `0002.png`, ...

### Dataset Quality Recommendations

For best reconstruction results:
- **Frame rate**: 10-30 fps (slower movement = better quality)
- **Image resolution**: 640×480 or higher
- **Depth range**: 0.1m to 20m (adjustable in config)
- **Camera movement**: Smooth, avoid sudden movements
- **Lighting**: Consistent, avoid over/under-exposure
- **Coverage**: Multiple viewpoints of each surface

## Contact & Support

For questions about data format or issues with datasets:
- Open an issue on GitHub
- Check the main README.md for more documentation
- Refer to QUICKSTART.md for step-by-step guide

---

**Last Updated**: 2025-12-23
