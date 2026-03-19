"""Export labeled point clouds to PLY and LAS formats.

These exporters write per-point labels as scalar/classification fields
so that downstream tools can consume the annotation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def export_labeled_ply(
    coord: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    colors: Optional[np.ndarray] = None,
) -> Path:
    """Export a labeled point cloud to PLY with a ``label`` scalar field.

    Parameters
    ----------
    coord : np.ndarray
        (N, 3) XYZ coordinates.
    labels : np.ndarray
        (N,) integer labels for each point.
    output_path : str or Path
        Destination .ply file path.
    colors : np.ndarray, optional
        (N, 3) uint8 RGB colors.  If provided, they are written to the PLY.

    Returns
    -------
    Path
        The written file path.
    """
    import open3d as o3d

    output_path = Path(output_path)
    n = coord.shape[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    # Write PLY, then append the label field via manual editing.
    # Open3D doesn't natively support custom scalar fields in PLY,
    # so we write a basic PLY and then patch the header + data.
    _write_ply_with_labels(output_path, coord, labels, colors)
    return output_path


def export_labeled_las(
    coord: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    colors: Optional[np.ndarray] = None,
) -> Path:
    """Export a labeled point cloud to LAS with the classification field.

    Parameters
    ----------
    coord : np.ndarray
        (N, 3) XYZ coordinates.
    labels : np.ndarray
        (N,) integer labels for each point.
    output_path : str or Path
        Destination .las file path.
    colors : np.ndarray, optional
        (N, 3) uint8 RGB colors.

    Returns
    -------
    Path
        The written file path.
    """
    import laspy

    output_path = Path(output_path)

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = [0.0001, 0.0001, 0.0001]
    header.offsets = [
        float(coord[:, 0].mean()),
        float(coord[:, 1].mean()),
        float(coord[:, 2].mean()),
    ]
    las = laspy.LasData(header)

    las.x = coord[:, 0]
    las.y = coord[:, 1]
    las.z = coord[:, 2]

    # Classification field (uint8, max 255)
    las.classification = labels.astype(np.uint8)

    if colors is not None:
        # LAS point format 2 has RGB; scale uint8 to uint16
        las.red = colors[:, 0].astype(np.uint16) * 257
        las.green = colors[:, 1].astype(np.uint16) * 257
        las.blue = colors[:, 2].astype(np.uint16) * 257

    las.write(str(output_path))
    return output_path


def _write_ply_with_labels(
    path: Path,
    coord: np.ndarray,
    labels: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Write a PLY file with an extra ``label`` scalar field."""
    n = coord.shape[0]
    has_color = colors is not None

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines += [
        "property int label",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))

        for i in range(n):
            # XYZ as float32
            f.write(np.float32(coord[i, 0]).tobytes())
            f.write(np.float32(coord[i, 1]).tobytes())
            f.write(np.float32(coord[i, 2]).tobytes())
            if has_color:
                f.write(np.uint8(colors[i, 0]).tobytes())
                f.write(np.uint8(colors[i, 1]).tobytes())
                f.write(np.uint8(colors[i, 2]).tobytes())
            f.write(np.int32(labels[i]).tobytes())
