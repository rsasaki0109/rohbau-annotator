"""Tests for .npy loading functions."""

import numpy as np
import pytest

from rohbau_annotator.loader import Rohbau3DScan, load_labels, load_scan


@pytest.fixture()
def scan_dir(tmp_path):
    """Create a minimal scan directory with required .npy files."""
    n_points = 100
    np.save(tmp_path / "coord.npy", np.random.randn(n_points, 3).astype(np.float64))
    np.save(tmp_path / "color.npy", np.random.randint(0, 256, (n_points, 3)).astype(np.uint8))
    np.save(tmp_path / "intensity.npy", np.random.rand(n_points, 1).astype(np.float64))
    np.save(tmp_path / "normal.npy", np.random.randn(n_points, 3).astype(np.float64))
    return tmp_path


@pytest.fixture()
def scan_dir_with_images(scan_dir):
    """Scan directory with panorama images."""
    from PIL import Image

    h, w = 64, 128
    Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)).save(
        scan_dir / "img_color.png"
    )
    return scan_dir


class TestLoadScan:
    def test_load_minimal_scan(self, scan_dir):
        scan = load_scan(scan_dir)
        assert isinstance(scan, Rohbau3DScan)
        assert scan.num_points == 100
        assert scan.coord.shape == (100, 3)
        assert scan.color.shape == (100, 3)
        assert scan.intensity.shape == (100, 1)
        assert scan.normal.shape == (100, 3)

    def test_panorama_images_none_when_missing(self, scan_dir):
        scan = load_scan(scan_dir)
        assert scan.img_color is None
        assert scan.img_depth is None
        assert scan.panorama_shape is None

    def test_panorama_images_loaded(self, scan_dir_with_images):
        scan = load_scan(scan_dir_with_images)
        assert scan.img_color is not None
        assert scan.panorama_shape == (64, 128)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scan(tmp_path / "nonexistent")

    def test_missing_required_file_raises(self, tmp_path):
        """A directory missing coord.npy should raise."""
        tmp_path.mkdir(exist_ok=True)
        np.save(tmp_path / "color.npy", np.zeros((10, 3)))
        np.save(tmp_path / "intensity.npy", np.zeros((10, 1)))
        np.save(tmp_path / "normal.npy", np.zeros((10, 3)))
        with pytest.raises(FileNotFoundError, match="coord.npy"):
            load_scan(tmp_path)

    def test_scan_dir_is_path(self, scan_dir):
        scan = load_scan(str(scan_dir))
        assert scan.scan_dir == scan_dir


class TestLoadLabels:
    def test_load_labels(self, tmp_path):
        labels = np.array([0, 1, 2, 3, 1], dtype=np.int32)
        path = tmp_path / "labels.npy"
        np.save(path, labels)
        loaded = load_labels(path)
        np.testing.assert_array_equal(loaded, labels)

    def test_missing_label_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labels(tmp_path / "missing.npy")
