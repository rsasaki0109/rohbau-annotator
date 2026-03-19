"""Tests for point cloud exporters (PLY and LAS)."""

import numpy as np
import pytest

from rohbau_annotator.exporters import export_labeled_ply, export_labeled_las


@pytest.fixture()
def sample_data():
    """Generate sample point cloud with labels."""
    np.random.seed(42)
    n = 30
    coord = np.random.randn(n, 3).astype(np.float64)
    labels = np.random.randint(0, 5, n).astype(np.int32)
    colors = np.random.randint(0, 256, (n, 3)).astype(np.uint8)
    return coord, labels, colors


class TestExportLabeledPly:
    def test_creates_file(self, tmp_path, sample_data):
        coord, labels, colors = sample_data
        out = tmp_path / "labeled.ply"
        result = export_labeled_ply(coord, labels, out, colors=colors)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_without_colors(self, tmp_path, sample_data):
        coord, labels, _ = sample_data
        out = tmp_path / "labeled_nocolor.ply"
        result = export_labeled_ply(coord, labels, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_ply_header_contains_label(self, tmp_path, sample_data):
        coord, labels, colors = sample_data
        out = tmp_path / "labeled.ply"
        export_labeled_ply(coord, labels, out, colors=colors)
        with open(out, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
        header_str = header.decode("ascii")
        assert "property int label" in header_str
        assert "property float x" in header_str

    def test_roundtrip_via_open3d(self, tmp_path, sample_data):
        """Open3D should be able to read the PLY (ignoring the label field)."""
        import open3d as o3d

        coord, labels, colors = sample_data
        out = tmp_path / "labeled.ply"
        export_labeled_ply(coord, labels, out, colors=colors)
        pcd = o3d.io.read_point_cloud(str(out))
        assert len(pcd.points) == len(coord)


class TestExportLabeledLas:
    def test_creates_file(self, tmp_path, sample_data):
        coord, labels, colors = sample_data
        out = tmp_path / "labeled.las"
        result = export_labeled_las(coord, labels, out, colors=colors)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_without_colors(self, tmp_path, sample_data):
        coord, labels, _ = sample_data
        out = tmp_path / "labeled_nocolor.las"
        result = export_labeled_las(coord, labels, out)
        assert out.exists()

    def test_roundtrip_labels(self, tmp_path, sample_data):
        """Labels written should be readable back via laspy."""
        import laspy

        coord, labels, colors = sample_data
        out = tmp_path / "labeled.las"
        export_labeled_las(coord, labels, out, colors=colors)

        las = laspy.read(str(out))
        read_labels = np.asarray(las.classification)
        np.testing.assert_array_equal(read_labels, labels.astype(np.uint8))

    def test_roundtrip_coordinates(self, tmp_path, sample_data):
        """Coordinates should survive a write/read cycle."""
        import laspy

        coord, labels, _ = sample_data
        out = tmp_path / "labeled.las"
        export_labeled_las(coord, labels, out)

        las = laspy.read(str(out))
        read_coord = np.column_stack([las.x, las.y, las.z])
        np.testing.assert_allclose(read_coord, coord, atol=1e-3)
