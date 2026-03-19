"""Load point cloud data from Rohbau3D .npy directories or single files (PLY/PCD/LAS).

Rohbau3D data layout per scan directory:
    coord.npy       (N, 3)  float64  XYZ coordinates
    color.npy       (N, 3)  uint8    RGB color
    intensity.npy   (N, 1)  float64  laser intensity
    normal.npy      (N, 3)  float64  surface normals
    img_color.png            panorama color image
    img_depth.png            panorama depth image
    img_intensity.png        panorama intensity image
    img_normal.png           panorama normal image

Single-file formats (PLY, PCD, LAS/LAZ) are also supported via the importers module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class Rohbau3DScan:
    """A loaded point cloud scan with optional panorama images.

    Works with both Rohbau3D .npy directories and single-file point clouds.
    """

    scan_dir: Path
    coord: np.ndarray           # (N, 3) float64
    color: np.ndarray           # (N, 3) uint8
    intensity: np.ndarray       # (N, 1) float64
    normal: np.ndarray          # (N, 3) float64
    source_file: Optional[Path] = None  # set when loaded from a single file
    img_color: Optional[np.ndarray] = None       # (H, W, 3) uint8
    img_depth: Optional[np.ndarray] = None       # (H, W) or (H, W, 3)
    img_intensity: Optional[np.ndarray] = None   # (H, W) or (H, W, 3)
    img_normal: Optional[np.ndarray] = None      # (H, W, 3)

    @property
    def num_points(self) -> int:
        return self.coord.shape[0]

    @property
    def panorama_shape(self) -> Optional[tuple]:
        if self.img_color is not None:
            return self.img_color.shape[:2]
        return None


def load_scan(path: str | Path) -> Rohbau3DScan:
    """Load a point cloud scan from a directory or single file.

    Parameters
    ----------
    path : str or Path
        Path to either:
        - A Rohbau3D scan directory containing .npy files and panorama images.
        - A single point cloud file (.ply, .pcd, .las, .laz).

    Returns
    -------
    Rohbau3DScan
        Loaded scan data.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or required files are missing.
    ValueError
        If a single file has an unsupported extension.
    """
    path = Path(path)

    if path.is_dir():
        return _load_npy_directory(path)
    elif path.is_file():
        return _load_single_file(path)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def _load_npy_directory(scan_dir: Path) -> Rohbau3DScan:
    """Load a Rohbau3D scan from a directory of .npy files."""
    # Required point cloud arrays
    coord_path = scan_dir / "coord.npy"
    color_path = scan_dir / "color.npy"
    intensity_path = scan_dir / "intensity.npy"
    normal_path = scan_dir / "normal.npy"

    for p in [coord_path, color_path, intensity_path, normal_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    coord = np.load(coord_path)
    color = np.load(color_path)
    intensity = np.load(intensity_path)
    normal = np.load(normal_path)

    # Optional panorama images
    img_color = _load_image(scan_dir / "img_color.png")
    img_depth = _load_image(scan_dir / "img_depth.png")
    img_intensity = _load_image(scan_dir / "img_intensity.png")
    img_normal = _load_image(scan_dir / "img_normal.png")

    return Rohbau3DScan(
        scan_dir=scan_dir,
        coord=coord,
        color=color,
        intensity=intensity,
        normal=normal,
        img_color=img_color,
        img_depth=img_depth,
        img_intensity=img_intensity,
        img_normal=img_normal,
    )


def _load_single_file(file_path: Path) -> Rohbau3DScan:
    """Load a scan from a single PLY/PCD/LAS file."""
    from rohbau_annotator.importers import auto_import, is_supported_file

    if not is_supported_file(file_path):
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            "Use a Rohbau3D directory or a PLY/PCD/LAS file."
        )

    coord, color, intensity, normal = auto_import(file_path)

    return Rohbau3DScan(
        scan_dir=file_path.parent,
        coord=coord,
        color=color,
        intensity=intensity,
        normal=normal,
        source_file=file_path,
    )


def _load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image file as a numpy array, or return None if it doesn't exist."""
    if not path.exists():
        return None
    img = Image.open(path)
    return np.array(img)


def load_labels(label_path: str | Path) -> np.ndarray:
    """Load a label .npy file.

    Parameters
    ----------
    label_path : str or Path
        Path to a label .npy file of shape (N,) with integer class IDs.

    Returns
    -------
    np.ndarray
        Integer label array of shape (N,).
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    return np.load(label_path)
