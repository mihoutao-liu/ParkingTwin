# Quick Start Guide

This is a quick reference guide for using ParkingTwin. For detailed documentation, see [README.md](README.md).

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare Your Dataset

### Option A: Download Pre-configured Dataset (Recommended)

📥 **Download Complete Dataset**:
- Quark Cloud Drive: https://pan.quark.cn/s/ab5c6a9fa61d
- Extraction Code: `VC1Y`

Includes RGB, Depth, Pose, First Frame Pose, and Camera Intrinsics.

For more download options (TSDF models, split blocks, etc.), see [Datasets/README.md](Datasets/README.md).

### Option B: Use Your Own Dataset

Create a folder structure:

```
Datasets/your_dataset/
├── pose/                  # Camera poses (0000.txt, 0001.txt, ...)
├── first_pose_osm.txt    # First frame pose
├── color/                # RGB images (0000.png, 0001.png, ...)
├── depth/                # Depth images (0000.png, 0001.png, ...)
├── K_rectified.npz       # Camera intrinsics
└── mesh.ply              # 3D mesh (from TSDF download)
```

## 3. Configure Paths

Edit `configs/config_default.yaml`:

```yaml
paths:
  mesh_path: "Datasets/your_dataset/mesh.ply"
  pose_dir: "Datasets/your_dataset/pose"
  first_pose_txt: "Datasets/your_dataset/first_pose.txt"
  K_npz: "Datasets/your_dataset/K_rectified.npz"
  rgb_dir: "Datasets/your_dataset/color"
  depth_dir: "Datasets/your_dataset/depth"
```

## 4. Select First Frame Pose (Optional)

```bash
python scripts/osm_pose_selector.py Datasets/ICPARKOSM/ICPARK.osm Datasets/your_dataset/pose
```

Press `s` to save the pose to `first_pose.txt`.

## 5. Generate Mesh from OSM (Optional)

```bash
python scripts/Osm2Tsdf.py \
  --osm Datasets/ICPARKOSM/ICPARK.osm \
  --outdir output/ICPARKOSM_generated \
  --voxel 0.10 --height 3.0 --trunc 0.40
```

Update `mesh_path` in config to point to the generated mesh.

## 6. Run Texture Mapping

```bash
python scripts/texture_realtime.py --config configs/config_default.yaml
```

Press `ENTER` in the visualization window to start texturing. The result will be saved to `output/textured_meshes/`.

