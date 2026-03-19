"""Spherical (equirectangular) projection between 3D points and 2D panorama pixels.

Rohbau3D uses equirectangular spherical projection:
    azimuth   = atan2(y, x)          in [-pi, pi]
    elevation = arcsin(z / r)        in [-pi/2, pi/2]
    r         = sqrt(x^2 + y^2 + z^2)

Panorama image coordinate mapping:
    u = (pi - azimuth) / (2 * pi) * W        column  [0, W)
    v = (pi/2 - elevation) / pi * H          row     [0, H)

Back-projection from (u, v, depth) to 3D:
    azimuth   = pi - u / W * 2 * pi
    elevation = pi/2 - v / H * pi
    x = depth * cos(elevation) * cos(azimuth)
    y = depth * cos(elevation) * sin(azimuth)
    z = depth * sin(elevation)
"""

from __future__ import annotations

import numpy as np


def points_to_panorama_uv(
    points: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to panorama pixel coordinates.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    img_h : int
        Panorama image height in pixels.
    img_w : int
        Panorama image width in pixels.

    Returns
    -------
    u : np.ndarray
        (N,) column indices (float, may need rounding/clipping).
    v : np.ndarray
        (N,) row indices (float, may need rounding/clipping).
    r : np.ndarray
        (N,) radial distance from origin.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.maximum(r, 1e-10)

    azimuth = np.arctan2(y, x)                # [-pi, pi]
    elevation = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))  # [-pi/2, pi/2]

    u = (np.pi - azimuth) / (2.0 * np.pi) * img_w
    v = (np.pi / 2.0 - elevation) / np.pi * img_h

    return u, v, r


def points_to_panorama_indices(
    points: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to integer panorama pixel coordinates.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    img_h : int
        Panorama image height in pixels.
    img_w : int
        Panorama image width in pixels.

    Returns
    -------
    rows : np.ndarray
        (N,) integer row indices, clipped to [0, img_h - 1].
    cols : np.ndarray
        (N,) integer column indices, clipped to [0, img_w - 1].
    r : np.ndarray
        (N,) radial distances.
    """
    u, v, r = points_to_panorama_uv(points, img_h, img_w)

    cols = np.clip(np.round(u).astype(np.int64), 0, img_w - 1)
    rows = np.clip(np.round(v).astype(np.int64), 0, img_h - 1)

    return rows, cols, r


def panorama_uv_to_points(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Back-project panorama pixels to 3D coordinates.

    Parameters
    ----------
    u : np.ndarray
        (M,) column indices (pixel coordinates).
    v : np.ndarray
        (M,) row indices (pixel coordinates).
    depth : np.ndarray
        (M,) radial depth for each pixel.
    img_h : int
        Panorama image height in pixels.
    img_w : int
        Panorama image width in pixels.

    Returns
    -------
    points : np.ndarray
        (M, 3) array of XYZ coordinates.
    """
    azimuth = np.pi - (u / img_w) * 2.0 * np.pi
    elevation = np.pi / 2.0 - (v / img_h) * np.pi

    cos_elev = np.cos(elevation)
    x = depth * cos_elev * np.cos(azimuth)
    y = depth * cos_elev * np.sin(azimuth)
    z = depth * np.sin(elevation)

    return np.stack([x, y, z], axis=-1)


def build_panorama_label_map(
    label_2d: np.ndarray,
    points: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Transfer 2D panorama labels to 3D point cloud labels.

    For each 3D point, project it to the panorama and look up the label.

    Parameters
    ----------
    label_2d : np.ndarray
        (H, W) integer label map on the panorama image.
    points : np.ndarray
        (N, 3) array of XYZ point cloud coordinates.
    img_h : int
        Panorama image height (must match label_2d.shape[0]).
    img_w : int
        Panorama image width (must match label_2d.shape[1]).

    Returns
    -------
    labels_3d : np.ndarray
        (N,) integer labels for each point.
    """
    rows, cols, _ = points_to_panorama_indices(points, img_h, img_w)
    labels_3d = label_2d[rows, cols]
    return labels_3d


def render_label_panorama(
    points: np.ndarray,
    labels: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Render a label panorama from labeled 3D points.

    Uses z-buffering: for pixels with multiple projected points,
    the closest point's label is used.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    labels : np.ndarray
        (N,) integer labels.
    img_h : int
        Output panorama height.
    img_w : int
        Output panorama width.

    Returns
    -------
    label_map : np.ndarray
        (H, W) integer label panorama, 0 for unlabeled pixels.
    """
    rows, cols, r = points_to_panorama_indices(points, img_h, img_w)

    label_map = np.zeros((img_h, img_w), dtype=np.int32)
    depth_buf = np.full((img_h, img_w), np.inf, dtype=np.float64)

    for i in range(len(points)):
        row, col, dist = rows[i], cols[i], r[i]
        if dist < depth_buf[row, col]:
            depth_buf[row, col] = dist
            label_map[row, col] = labels[i]

    return label_map
