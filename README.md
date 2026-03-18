# rohbau-annotator

Annotation tool for [Rohbau3D](https://github.com/2KangHo/Rohbau3D) construction site point clouds.
Annotate on 2D panorama images and back-project labels to 3D point clouds using equirectangular spherical projection.

## Semantic Classes

wall, floor, ceiling, column, beam, door, window, pipe, duct, cable_tray, rebar, formwork, other

## Installation

```bash
pip install -e .
```

## Usage

### Annotate a scan

```bash
rohbau-annotator annotate /path/to/scan_dir
```

Opens a matplotlib window showing the panorama image. Paint regions with the selected class label using the mouse. Close the window to save `label_2d.npy` and `label_3d.npy` in the scan directory.

### Export 3D labels from existing 2D annotations

```bash
rohbau-annotator export /path/to/scan_dir
```

### View annotation statistics

```bash
rohbau-annotator stats /path/to/scan_dir
```

## Data Format

Each scan directory must contain:

| File | Shape | Description |
|------|-------|-------------|
| `coord.npy` | (N, 3) | XYZ coordinates |
| `color.npy` | (N, 3) | RGB color |
| `intensity.npy` | (N, 1) | Laser intensity |
| `normal.npy` | (N, 3) | Surface normals |
| `img_color.png` | (H, W, 3) | Panorama color image |

## Projection

Uses equirectangular spherical projection:

```
azimuth   = atan2(y, x)
elevation = arcsin(z / r)
u = (pi - azimuth) / (2*pi) * W
v = (pi/2 - elevation) / pi * H
```

## License

MIT
