"""Load Rohbau3D .npy files and panorama images.

Rohbau3D data layout per scan directory:
    coord.npy       (N, 3)  float64  XYZ coordinates
    color.npy       (N, 3)  uint8    RGB color
    intensity.npy   (N, 1)  float64  laser intensity
    normal.npy      (N, 3)  float64  surface normals
    img_color.png            panorama color image
    img_depth.png            panorama depth image
    img_intensity.png        panorama intensity image
    img_normal.png           panorama normal image
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class Rohbau3DScan:
    """A single Rohbau3D scan with point cloud data and panorama images."""

    scan_dir: Path
    coord: np.ndarray           # (N, 3) float64
    color: np.ndarray           # (N, 3) uint8
    intensity: np.ndarray       # (N, 1) float64
    normal: np.ndarray          # (N, 3) float64
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


def load_scan(scan_dir: str | Path) -> Rohbau3DScan:
    """Load a Rohbau3D scan from a directory.

    Parameters
    ----------
    scan_dir : str or Path
        Path to a scan directory containing .npy files and panorama images.

    Returns
    -------
    Rohbau3DScan
        Loaded scan data.

    Raises
    ------
    FileNotFoundError
        If required .npy files are missing.
    """
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan directory not found: {scan_dir}")

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
