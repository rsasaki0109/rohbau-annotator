"""Tests for CLI commands using CliRunner (non-GUI commands: export, stats)."""

import numpy as np
import pytest
from click.testing import CliRunner

from rohbau_annotator.cli import main


@pytest.fixture()
def scan_dir(tmp_path):
    """Create a minimal scan directory with required .npy files."""
    n_points = 50
    np.save(tmp_path / "coord.npy", np.random.randn(n_points, 3).astype(np.float64))
    np.save(tmp_path / "color.npy", np.random.randint(0, 256, (n_points, 3)).astype(np.uint8))
    np.save(tmp_path / "intensity.npy", np.random.rand(n_points, 1).astype(np.float64))
    np.save(tmp_path / "normal.npy", np.random.randn(n_points, 3).astype(np.float64))
    return tmp_path


@pytest.fixture()
def scan_dir_with_labels(scan_dir):
    """Scan directory with existing 2D and 3D label files."""
    h, w = 64, 128
    label_2d = np.zeros((h, w), dtype=np.int32)
    # Paint a region as class 1 (wall)
    label_2d[10:30, 20:60] = 1
    # Paint another region as class 2 (floor)
    label_2d[30:50, 20:60] = 2
    np.save(scan_dir / "label_2d.npy", label_2d)

    label_3d = np.zeros(50, dtype=np.int32)
    label_3d[:20] = 1
    label_3d[20:35] = 2
    np.save(scan_dir / "label_3d.npy", label_3d)
    return scan_dir


class TestVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestStatsCommand:
    def test_stats_no_labels(self, scan_dir):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(scan_dir)])
        assert result.exit_code == 0
        assert "No label_2d.npy found" in result.output
        assert "No label_3d.npy found" in result.output

    def test_stats_with_labels(self, scan_dir_with_labels):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(scan_dir_with_labels)])
        assert result.exit_code == 0
        assert "2D label map:" in result.output
        assert "Labeled pixels:" in result.output
        assert "wall" in result.output
        assert "floor" in result.output
        assert "3D labels:" in result.output
        assert "Labeled points:" in result.output

    def test_stats_nonexistent_dir(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", "/nonexistent/path"])
        assert result.exit_code != 0


class TestExportCommand:
    def test_export_no_label_2d(self, scan_dir):
        """Export without label_2d.npy should fail."""
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(scan_dir)])
        assert result.exit_code != 0
        assert "label_2d.npy" in result.output

    def test_export_success(self, scan_dir_with_labels):
        runner = CliRunner()
        out_path = scan_dir_with_labels / "exported_labels.npy"
        result = runner.invoke(
            main, ["export", str(scan_dir_with_labels), "-o", str(out_path)]
        )
        assert result.exit_code == 0
        assert "Saved 3D labels" in result.output
        exported = np.load(out_path)
        assert exported.shape == (50,)

    def test_export_default_output(self, scan_dir_with_labels):
        """Export without -o should write to label_3d.npy in scan dir."""
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(scan_dir_with_labels)])
        assert result.exit_code == 0
        label_3d = np.load(scan_dir_with_labels / "label_3d.npy")
        assert label_3d.shape == (50,)

    def test_export_nonexistent_dir(self):
        runner = CliRunner()
        result = runner.invoke(main, ["export", "/nonexistent/path"])
        assert result.exit_code != 0


class TestHelpText:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "annotate" in result.output
        assert "export" in result.output
        assert "stats" in result.output
