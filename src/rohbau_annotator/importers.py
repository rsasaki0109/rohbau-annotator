"""Import point clouds from PLY, PCD, and LAS/LAZ formats.

All importers return the same tuple format used by Rohbau3D .npy loading:
    coord      (N, 3)  float64   XYZ coordinates
    color      (N, 3)  uint8     RGB color  (zeros if absent)
    intensity  (N, 1)  float64   laser intensity (zeros if absent)
    normal     (N, 3)  float64   surface normals (zeros if absent)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Type alias for the standard point-cloud tuple
PointCloudArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

_SUPPORTED_EXTENSIONS = {".ply", ".pcd", ".las", ".laz"}


def import_ply(path: str | Path) -> PointCloudArrays:
    """Import a PLY point cloud file using Open3D.

    Parameters
    ----------
    path : str or Path
        Path to a .ply file.

    Returns
    -------
    coord : np.ndarray (N, 3) float64
    color : np.ndarray (N, 3) uint8
    intensity : np.ndarray (N, 1) float64
    normal : np.ndarray (N, 3) float64
    """
    import open3d as o3d

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    return _open3d_to_arrays(pcd)


def import_pcd(path: str | Path) -> PointCloudArrays:
    """Import a PCD point cloud file using Open3D.

    Parameters
    ----------
    path : str or Path
        Path to a .pcd file.

    Returns
    -------
    coord : np.ndarray (N, 3) float64
    color : np.ndarray (N, 3) uint8
    intensity : np.ndarray (N, 1) float64
    normal : np.ndarray (N, 3) float64
    """
    import open3d as o3d

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCD file not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    return _open3d_to_arrays(pcd)


def import_las(path: str | Path) -> PointCloudArrays:
    """Import a LAS/LAZ point cloud file using laspy.

    Parameters
    ----------
    path : str or Path
        Path to a .las or .laz file.

    Returns
    -------
    coord : np.ndarray (N, 3) float64
    color : np.ndarray (N, 3) uint8
    intensity : np.ndarray (N, 1) float64
    normal : np.ndarray (N, 3) float64
    """
    import laspy

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = laspy.read(str(path))
    n = len(las.points)

    coord = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    # Color: LAS stores RGB as uint16; scale to uint8
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        r = np.asarray(las.red, dtype=np.float64)
        g = np.asarray(las.green, dtype=np.float64)
        b = np.asarray(las.blue, dtype=np.float64)
        # LAS 1.2+ uses 16-bit color; scale to 8-bit
        max_val = max(r.max(), g.max(), b.max(), 1.0)
        if max_val > 255:
            r = (r / 65535.0 * 255.0)
            g = (g / 65535.0 * 255.0)
            b = (b / 65535.0 * 255.0)
        color = np.column_stack([r, g, b]).astype(np.uint8)
    else:
        color = np.zeros((n, 3), dtype=np.uint8)

    # Intensity
    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64).reshape(-1, 1)
    else:
        intensity = np.zeros((n, 1), dtype=np.float64)

    # LAS does not store normals
    normal = np.zeros((n, 3), dtype=np.float64)

    return coord, color, intensity, normal


def auto_import(path: str | Path) -> PointCloudArrays:
    """Detect point cloud format by extension and import.

    Supported formats: .ply, .pcd, .las, .laz

    Parameters
    ----------
    path : str or Path
        Path to a point cloud file.

    Returns
    -------
    coord, color, intensity, normal : same as individual importers.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".ply":
        return import_ply(path)
    elif ext == ".pcd":
        return import_pcd(path)
    elif ext in (".las", ".laz"):
        return import_las(path)
    else:
        raise ValueError(
            f"Unsupported point cloud format: '{ext}'. "
            f"Supported formats: {sorted(_SUPPORTED_EXTENSIONS)}"
        )


def is_supported_file(path: str | Path) -> bool:
    """Check whether a path has a supported point cloud file extension."""
    return Path(path).suffix.lower() in _SUPPORTED_EXTENSIONS


def _open3d_to_arrays(pcd) -> PointCloudArrays:
    """Convert an Open3D PointCloud to the standard array tuple."""
    coord = np.asarray(pcd.points, dtype=np.float64)
    n = coord.shape[0]

    if pcd.has_colors():
        color = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        color = np.zeros((n, 3), dtype=np.uint8)

    # Open3D PointCloud doesn't natively store intensity as a field;
    # for PLY/PCD files with intensity, it may be absent.
    intensity = np.zeros((n, 1), dtype=np.float64)

    if pcd.has_normals():
        normal = np.asarray(pcd.normals, dtype=np.float64)
    else:
        normal = np.zeros((n, 3), dtype=np.float64)

    return coord, color, intensity, normal
