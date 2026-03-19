# rohbau-annotator

[![CI](https://github.com/rsasaki0109/rohbau-annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/rohbau-annotator/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Universal TLS Point Cloud Annotator with SAM2.

Annotate on 2D panorama images and back-project labels to 3D point clouds using equirectangular spherical projection. Supports **PLY**, **PCD**, **LAS/LAZ** single files as well as [Rohbau3D](https://github.com/2KangHo/Rohbau3D) `.npy` directories.

## Supported Input Formats

| Format | Extension | Library |
|--------|-----------|---------|
| PLY | `.ply` | Open3D |
| PCD | `.pcd` | Open3D |
| LAS/LAZ | `.las`, `.laz` | laspy |
| Rohbau3D | directory with `.npy` files | NumPy |

## Workflow

```
     ┌──────────────────────────┐
     │  Point Cloud Input       │
     │  PLY / PCD / LAS / .npy  │
     └────────────┬─────────────┘
                  │
     ┌────────────▼────────────┐
     │  Equirectangular         │
     │  Spherical Projection    │
     │  (3D → 2D panorama)      │
     └────────────┬─────────────┘
                  │
     ┌────────────▼────────────┐
     │  Interactive 2D          │
     │  Annotation (matplotlib) │
     │  + optional SAM2         │
     │  → label_2d.npy          │
     └────────────┬─────────────┘
                  │
     ┌────────────▼────────────┐
     │  Back-projection         │
     │  (2D → 3D labels)        │
     │  → .npy / .ply / .las    │
     └──────────────────────────┘
```

## Installation

```bash
pip install -e .
```

For development (includes pytest and ruff):

```bash
pip install -e ".[dev]"
```

For SAM2 semi-automatic segmentation support:

```bash
pip install -e ".[sam2]"
```

You also need a SAM2 checkpoint file (e.g., `sam2_hiera_tiny.pt`).
Download from the [SAM2 repository](https://github.com/facebookresearch/segment-anything-2).

## Usage

### Annotate a Rohbau3D scan directory

```bash
rohbau-annotator annotate /path/to/scan_dir
```

### Annotate a PLY file

```bash
rohbau-annotator annotate /path/to/scan.ply
```

### Annotate a PCD file

```bash
rohbau-annotator annotate /path/to/scan.pcd
```

### Annotate a LAS file

```bash
rohbau-annotator annotate /path/to/scan.las
```

Opens a matplotlib window showing the panorama image. Paint regions with the selected class label using the mouse. Close the window to save `label_2d.npy` and `label_3d.npy` alongside the input.

### Export 3D labels from existing 2D annotations

```bash
# Default: export as .npy
rohbau-annotator export /path/to/scan_dir

# Export as labeled PLY with per-point label field
rohbau-annotator export /path/to/scan.ply -f ply -o labeled_output.ply

# Export as LAS with classification field
rohbau-annotator export /path/to/scan.las -f las -o labeled_output.las
```

### Annotate with SAM2 assistance

```bash
rohbau-annotator annotate /path/to/scan.ply --sam2 --sam2-checkpoint /path/to/sam2_hiera_tiny.pt
```

In the annotation window, click the **SAM2** button to toggle SAM2 mode:

- **Left click** adds a foreground point (green `+`)
- **Right click** adds a background point (red `x`)
- SAM2 proposes a mask overlay after each click
- **Enter** accepts the mask (writes it to the label map with the selected class)
- **Escape** rejects the mask and clears all prompt points

### View annotation statistics

```bash
rohbau-annotator stats /path/to/scan_dir
```

## Projection Formula

The tool uses equirectangular spherical projection to map between 3D point cloud coordinates and 2D panorama pixel positions.

### Forward Projection (3D to 2D)

Given a 3D point `(x, y, z)`, compute spherical coordinates:

```
r         = sqrt(x² + y² + z²)
azimuth   = atan2(y, x)            ∈ [-π, π]
elevation = arcsin(z / r)          ∈ [-π/2, π/2]
```

Map to panorama pixel coordinates `(u, v)` for an image of size `W × H`:

```
u = (π - azimuth)   / (2π) × W    column ∈ [0, W)
v = (π/2 - elevation) / π  × H    row    ∈ [0, H)
```

### Back-projection (2D to 3D)

Given pixel coordinates `(u, v)` and depth `d`:

```
azimuth   = π   - (u / W) × 2π
elevation = π/2 - (v / H) × π

x = d × cos(elevation) × cos(azimuth)
y = d × cos(elevation) × sin(azimuth)
z = d × sin(elevation)
```

## Semantic Classes

The tool defines 13 annotatable semantic classes for construction site elements (plus unlabeled):

| ID | Name | Color |
|----|------|-------|
| 0 | unlabeled | (0, 0, 0) |
| 1 | wall | (174, 199, 232) |
| 2 | floor | (152, 223, 138) |
| 3 | ceiling | (31, 119, 180) |
| 4 | column | (255, 187, 120) |
| 5 | beam | (188, 189, 34) |
| 6 | door | (140, 86, 75) |
| 7 | window | (255, 152, 150) |
| 8 | pipe | (214, 39, 40) |
| 9 | duct | (197, 176, 213) |
| 10 | cable_tray | (148, 103, 189) |
| 11 | rebar | (196, 156, 148) |
| 12 | formwork | (23, 190, 207) |
| 13 | other | (127, 127, 127) |

## Data Format

### Rohbau3D directory

Each scan directory must contain:

| File | Shape | Description |
|------|-------|-------------|
| `coord.npy` | (N, 3) | XYZ coordinates |
| `color.npy` | (N, 3) | RGB color |
| `intensity.npy` | (N, 1) | Laser intensity |
| `normal.npy` | (N, 3) | Surface normals |
| `img_color.png` | (H, W, 3) | Panorama color image |

### Single-file point clouds

PLY, PCD, and LAS/LAZ files are loaded directly. Coordinates, colors, and normals are extracted where available; missing fields default to zeros.

## Running Tests

```bash
pytest -v tests/
```

## License

MIT
