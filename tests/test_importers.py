"""Tests for point cloud importers (PLY, PCD, LAS)."""

import numpy as np
import open3d as o3d
import pytest

from rohbau_annotator.importers import (
    auto_import,
    import_pcd,
    import_ply,
    is_supported_file,
)


@pytest.fixture()
def sample_points():
    """Generate sample point cloud data."""
    np.random.seed(42)
    n = 50
    coord = np.random.randn(n, 3).astype(np.float64)
    colors = np.random.randint(0, 256, (n, 3)).astype(np.uint8)
    normals = np.random.randn(n, 3).astype(np.float64)
    return coord, colors, normals


@pytest.fixture()
def ply_file(tmp_path, sample_points):
    """Write a PLY file with points, colors, and normals."""
    coord, colors, normals = sample_points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    path = tmp_path / "test.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


@pytest.fixture()
def pcd_file(tmp_path, sample_points):
    """Write a PCD file with points and colors."""
    coord, colors, _ = sample_points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    path = tmp_path / "test.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


class TestImportPly:
    def test_basic_load(self, ply_file, sample_points):
        coord, color, intensity, normal = import_ply(ply_file)
        expected_coord = sample_points[0]
        assert coord.shape == expected_coord.shape
        assert coord.dtype == np.float64
        assert color.shape == (50, 3)
        assert color.dtype == np.uint8
        assert intensity.shape == (50, 1)
        assert normal.shape == (50, 3)
        np.testing.assert_allclose(coord, expected_coord, atol=1e-5)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_ply(tmp_path / "nonexistent.ply")


class TestImportPcd:
    def test_basic_load(self, pcd_file, sample_points):
        coord, color, intensity, normal = import_pcd(pcd_file)
        assert coord.shape == (50, 3)
        assert color.shape == (50, 3)
        assert intensity.shape == (50, 1)
        # PCD didn't have normals written
        assert normal.shape == (50, 3)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_pcd(tmp_path / "nonexistent.pcd")


class TestAutoImport:
    def test_auto_ply(self, ply_file):
        coord, color, intensity, normal = auto_import(ply_file)
        assert coord.shape[1] == 3
        assert color.shape[1] == 3

    def test_auto_pcd(self, pcd_file):
        coord, color, intensity, normal = auto_import(pcd_file)
        assert coord.shape[1] == 3

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            auto_import(bad_file)


class TestIsSupportedFile:
    def test_supported(self):
        assert is_supported_file("scan.ply") is True
        assert is_supported_file("scan.pcd") is True
        assert is_supported_file("scan.las") is True
        assert is_supported_file("scan.laz") is True

    def test_unsupported(self):
        assert is_supported_file("scan.xyz") is False
        assert is_supported_file("scan.npy") is False
        assert is_supported_file("scan.txt") is False
