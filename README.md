# ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins

This repository contains the open-source code for the ParkingTwin paper. It provides a complete pipeline for real-time texture mapping and 3D reconstruction of parking lots from OpenStreetMap (OSM) data.

## Project Structure

```
open_source_code/
├── configs/
│   └── config_default.yaml          # Configuration file (modify paths here)
├── Datasets/                        # Data directory
│   ├── your_dataset/                # Your dataset folder
│   │   ├── pose/                    # Camera poses (4x4 matrices, .txt files)
│   │   ├── first_pose.txt          # First frame pose (4x4 matrix)
│   │   ├── color/                   # RGB images (.png)
│   │   ├── depth/                   # Depth images (.png)
│   │   ├── K_rectified.npz         # Camera intrinsics
│   │   └── mesh.ply                # 3D mesh (generated or provided)
│   └── ICPARKOSM/
│       └── ICPARK.osm               # Example OSM file
├── scripts/
│   ├── osm_pose_selector.py         # Interactive pose selector
│   ├── Osm2Tsdf.py                 # OSM to TSDF conversion
│   └── texture_realtime.py         # Main texture mapping script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 📥 Download Datasets

**Complete Dataset** (RGB, Depth, Pose, First Frame Pose, Camera Intrinsics):
- **Quark Cloud Drive**: https://pan.quark.cn/s/ab5c6a9fa61d
- **Extraction Code**: `VC1Y`

**TSDF Models** (3D Mesh from OSM):
- Original TSDF: https://pan.quark.cn/s/d9c2be04b0f8 (Code: `mLfv`)
- Coarse-Grained Model: https://pan.quark.cn/s/d9c2be04b0f8 (Code: `mLfv`) ⚠️ May reduce clarity
- Split TSDF (4 blocks): https://pan.quark.cn/s/23a4c2e91556 (Code: `6zYP`)
- Merge Code: https://pan.quark.cn/s/e1abf39d61a8 (Code: `sj5f`)

For detailed download instructions and data format, see [Datasets/README.md](Datasets/README.md).

---

## Installation

### 1. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

**Optional GPU Support** (for faster processing):
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### 2. Prepare Dataset

Download the pre-configured dataset (see Download Datasets section above) or prepare your own dataset with the following structure:

```
Datasets/your_dataset/
├── pose/                    # Camera poses directory
│   ├── 0000.txt            # 4x4 transformation matrix (T_wc)
│   ├── 0001.txt
│   └── ...
├── first_pose_osm.txt      # First frame pose (4x4 matrix)
├── color/                   # RGB images
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── depth/                   # Depth images
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── K_rectified.npz         # Camera intrinsics (contains K, W, H)
└── mesh.ply                # 3D mesh (from TSDF download)
```

**Data Format Requirements:**

- **Pose files**: Each `.txt` file contains a 4×4 transformation matrix:
  ```
  r11 r12 r13 tx
  r21 r22 r23 ty
  r31 r32 r33 tz
  0   0   0   1
  ```
  This represents the camera pose in world coordinates (T_wc).

- **First pose**: Same format as above, represents the first frame's absolute pose.

- **Camera intrinsics**: NPZ file containing:
  - `K`: 3×3 camera intrinsic matrix
  - `W`, `H` or `width`, `height`: Image dimensions

- **Images**: 
  - RGB: PNG format
  - Depth: PNG format, 16-bit unsigned integer (depth in millimeters)

## Usage

### Step 1: Configure Paths

Edit `configs/config_default.yaml` and update the paths section:

```yaml
paths:
  mesh_path: "Datasets/your_dataset/mesh.ply"
  pose_dir: "Datasets/your_dataset/pose"
  first_pose_txt: "Datasets/your_dataset/first_pose.txt"
  K_npz: "Datasets/your_dataset/K_rectified.npz"
  rgb_dir: "Datasets/your_dataset/color"
  depth_dir: "Datasets/your_dataset/depth"
```

### Step 2: Select First Frame Pose (Optional)

If you need to select the first frame pose interactively on an OSM map:

```bash
python scripts/osm_pose_selector.py Datasets/ICPARKOSM/ICPARK.osm Datasets/your_dataset/pose
```

**Controls:**
- **Ctrl + Click + Drag**: Set camera position and orientation
- **c**: Copy pose matrix to clipboard
- **s**: Save pose to `first_pose.txt`
- **r**: Reset selection
- **g**: Toggle grid alignment
- **q**: Quit

The tool will help you visualize your trajectory on the OSM map and select an appropriate first frame pose.

### Step 3: Generate TSDF from OSM (Optional)

If you want to generate a mesh from OSM data instead of using an existing mesh:

```bash
python scripts/Osm2Tsdf.py \
  --osm Datasets/ICPARKOSM/ICPARK.osm \
  --outdir output/ICPARKOSM_generated \
  --voxel 0.10 \
  --height 3.0 \
  --trunc 0.40
```

**Parameters:**
- `--osm`: Path to OSM file
- `--outdir`: Output directory for TSDF and mesh
- `--voxel`: Voxel resolution in meters (default: 0.10)
- `--height`: Height in meters for 2.5D TSDF (default: 3.0)
- `--trunc`: Truncation distance in meters (default: 0.40)

**Output:**
- `tsdf.npz`: TSDF voxel data
- `tsdf_3d_mesh.ply`: 3D mesh file
- `world_from_osm.json`: World coordinate transformation parameters
- `boundary_config.txt`: Boundary configuration

After generation, update `mesh_path` in `config_default.yaml` to point to the generated mesh.

### Step 4: Run Texture Mapping

Run the main texture mapping script:

```bash
python scripts/texture_realtime.py --config configs/config_default.yaml
```

**Process:**

1. **Trajectory Verification Phase** (Interactive):
   - The system will display your trajectory and mesh
   - Use keyboard controls to adjust poses if needed:
     - `1/2`: Rotate around X axis (+90°/-90°)
     - `3/4`: Rotate around Y axis (+90°/-90°)
     - `5/6`: Rotate around Z axis (+90°/-90°)
     - `R`: Reset rotation
     - `L`: Lock position, enter orientation adjustment mode
     - `7/8/9/0/U/I`: Adjust orientation
     - `S`: Save rotation configuration
     - `ENTER`: Confirm and start texturing
     - `Q`: Quit

2. **Texture Mapping Phase** (Automatic):
   - Processes frames according to configuration
   - Updates mesh colors in real-time
   - Saves final textured mesh to `output/textured_meshes/`

**Output:**
- Textured mesh saved to `output/textured_meshes/textured_mesh_YYYYMMDD_HHMMSS.ply`

## Configuration

The `configs/config_default.yaml` file contains all configuration options. Key sections:

- **paths**: Data file paths (modify these for your dataset)
- **texturing**: Frame sampling and processing parameters
- **visualization**: Display settings
- **openmvs**: Depth consistency and enhancement parameters
- **vehicle_detection**: Vehicle detection and removal settings
- **gpu**: GPU acceleration settings
- **step1_quality_filter** through **step8_post_processing**: 8-step optimization pipeline

See the configuration file for detailed comments on each parameter.

## Troubleshooting

### Common Issues

1. **GPU not detected**: Install CuPy matching your CUDA version, or the system will use CPU mode automatically.

2. **Out of memory**: Reduce `max_images` or increase `frame_sample_rate` in the configuration.

3. **Poor texture quality**: Enable optimization steps in the configuration (steps 1-8).

4. **Missing vertices**: Enable `post_processing.fill_empty_vertices` in the configuration.

### Performance Tips

- Use GPU acceleration when available (3-6x speedup)
- Adjust `frame_sample_rate` based on trajectory density
- Enable quality filtering to skip poor frames
- Reduce visualization update rate for faster processing

## Citation

If you use this code in your research, please cite:

```bibtex
@article{LIU2026114,
title = {ParkingTwin: Training-free streaming 3D reconstruction for parking-lot digital twins},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {240},
pages = {114-129},
year = {2026},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2026.07.009},
url = {https://www.sciencedirect.com/science/article/pii/S0924271626003564},
author = {Xinhao Liu and Yu Wang and Xiansheng Guo and Gordon Owusu Boateng and Yu Cao and Haonan Si and Xingchen Guo and Nirwan Ansari},
keywords = {Parking lot digital twin, structural prior, Real-time 3D reconstruction, Dynamic vehicle removal, Texture fusion, TSDF},
abstract = {High-fidelity digital twins of parking lots provide essential environmental priors for path planning, collision detection, and perception system validation of AVP. However, constructing such robot-oriented twins faces a fundamental “trilemma” involving geometric ambiguity, environmental interference, and computational constraints: (1) The restricted and sparse forward-facing views of mobile platforms lead to geometric degeneration in traditional methods due to insufficient parallax; (2) Frequent dynamic occlusions (e.g., moving vehicles) and extreme lighting variations impede consistent texture fusion; and (3) Exis ting neural rendering methods rely on computationally expensive offline optimization, failing to meet the real-time streaming requirements of edge-side robotics. To address these challenges, we propose ParkingTwin, a training-free, lightweight, and streaming 3D reconstruction system. The methodological core lies in Structural-Prior-Driven Geometric Construction: We leverage a CAD-derived structural prior to directly generate a metric-consistent 3D Truncated Signed Distance Field (TSDF) for prior-available controlled deployments. This approach reframes blind geometric search as a prior-guided deterministic mapping process, substantially alleviating the ill-posedness caused by sparse views while avoiding costly geometric optimization overhead. Built on this explicit geometric backbone, ParkingTwin further incorporates Geometry-Aware Dynamic Filtering for transient occlusion suppression and Illumination-Robust Fusion in the CIELAB color space for appearance completion under severe lighting changes. Experiments demonstrate that our system achieves 30+ Frames Per Second (FPS) online streaming reconstruction on an entry-level GPU (GTX 1660). On a large-scale 68,000 m2 real-world dataset, our method achieves an Structural Similarity Index Measure (SSIM) of 0.87 (a 16.0% improvement), accelerates end-to-end processing by approximately 15×, and reduces video memory usage by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) methods that require high-end GPUs (RTX 4090D). The system outputs explicit triangular meshes directly compatible with Unity/Unreal Engine (UE) digital twin workflows, effectively serving as an automated asset generator for initializing parking lot Digital Twins. Please visit our project page for the latest updates: https://mihoutao-liu.github.io/ParkingTwin/.}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.
