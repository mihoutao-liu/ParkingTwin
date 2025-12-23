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
@article{parkingtwin2025,
  title={ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.
