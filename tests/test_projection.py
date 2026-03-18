"""Tests for spherical projection math (3D <-> 2D round trips)."""

import numpy as np
import pytest

from rohbau_annotator.projection import (
    build_panorama_label_map,
    panorama_uv_to_points,
    points_to_panorama_indices,
    points_to_panorama_uv,
    render_label_panorama,
)

IMG_H = 512
IMG_W = 1024


class TestPointsToPanoramaUV:
    """Test forward projection from 3D to panorama (u, v)."""

    def test_point_on_positive_x_axis(self):
        """A point on the +X axis: azimuth=0, elevation=0 -> center-ish of panorama."""
        pts = np.array([[5.0, 0.0, 0.0]])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        # azimuth = atan2(0, 5) = 0  ->  u = pi/(2*pi) * W = W/2
        assert r[0] == pytest.approx(5.0)
        assert u[0] == pytest.approx(IMG_W / 2.0)
        # elevation = arcsin(0/5) = 0  ->  v = (pi/2) / pi * H = H/2
        assert v[0] == pytest.approx(IMG_H / 2.0)

    def test_point_on_negative_x_axis(self):
        """A point on the -X axis: azimuth=pi -> u=0 (or u=W, wrapping)."""
        pts = np.array([[-3.0, 0.0, 0.0]])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        assert r[0] == pytest.approx(3.0)
        # azimuth = atan2(0, -3) = pi  ->  u = (pi - pi)/(2*pi)*W = 0
        assert u[0] == pytest.approx(0.0)

    def test_point_on_positive_y_axis(self):
        """A point on the +Y axis: azimuth=pi/2 -> u = W/4."""
        pts = np.array([[0.0, 4.0, 0.0]])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        assert r[0] == pytest.approx(4.0)
        # azimuth = pi/2  ->  u = (pi - pi/2)/(2*pi)*W = W/4
        assert u[0] == pytest.approx(IMG_W / 4.0)

    def test_point_straight_up(self):
        """A point at the top (z=1): elevation=pi/2 -> v=0."""
        pts = np.array([[0.0, 0.0, 1.0]])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        assert r[0] == pytest.approx(1.0)
        assert v[0] == pytest.approx(0.0)

    def test_point_straight_down(self):
        """A point at the bottom (z=-1): elevation=-pi/2 -> v=H."""
        pts = np.array([[0.0, 0.0, -1.0]])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        assert r[0] == pytest.approx(1.0)
        assert v[0] == pytest.approx(float(IMG_H))

    def test_batch_shape(self):
        """Multiple points should produce matching output shapes."""
        pts = np.random.randn(100, 3)
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        assert u.shape == (100,)
        assert v.shape == (100,)
        assert r.shape == (100,)

    def test_near_origin_does_not_crash(self):
        """Points near the origin should not cause division-by-zero."""
        pts = np.array([[1e-15, 1e-15, 1e-15]])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()


class TestRoundTrip:
    """Test that 3D -> 2D -> 3D round-trips recover the original point."""

    @pytest.mark.parametrize(
        "point",
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 4.0, 0.0],
            [1.0, 1.0, 1.0],
            [-2.0, 3.0, -1.0],
            [0.5, -0.5, 0.3],
        ],
    )
    def test_forward_then_back(self, point):
        """Project 3D->2D->3D and verify the reconstruction matches."""
        pts = np.array([point])
        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        reconstructed = panorama_uv_to_points(u, v, r, IMG_H, IMG_W)
        np.testing.assert_allclose(reconstructed, pts, atol=1e-10)

    def test_round_trip_batch_random(self):
        """Random batch round trip test."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-10, 10, (200, 3))
        # Exclude points too close to origin
        norms = np.linalg.norm(pts, axis=1)
        pts = pts[norms > 0.1]

        u, v, r = points_to_panorama_uv(pts, IMG_H, IMG_W)
        reconstructed = panorama_uv_to_points(u, v, r, IMG_H, IMG_W)
        np.testing.assert_allclose(reconstructed, pts, atol=1e-9)


class TestPointsToPanoramaIndices:
    """Test integer index projection."""

    def test_indices_are_clipped(self):
        """Returned indices must be within [0, H-1] and [0, W-1]."""
        pts = np.array([[0.0, 0.0, -100.0]])  # extreme bottom
        rows, cols, r = points_to_panorama_indices(pts, IMG_H, IMG_W)
        assert 0 <= rows[0] <= IMG_H - 1
        assert 0 <= cols[0] <= IMG_W - 1

    def test_dtype_is_integer(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        rows, cols, r = points_to_panorama_indices(pts, IMG_H, IMG_W)
        assert np.issubdtype(rows.dtype, np.integer)
        assert np.issubdtype(cols.dtype, np.integer)


class TestBuildPanoramaLabelMap:
    """Test label transfer from 2D panorama to 3D."""

    def test_labels_shape_matches_points(self):
        label_2d = np.ones((IMG_H, IMG_W), dtype=np.int32) * 3
        pts = np.random.randn(50, 3)
        labels_3d = build_panorama_label_map(label_2d, pts, IMG_H, IMG_W)
        assert labels_3d.shape == (50,)

    def test_uniform_label_map(self):
        """All-same label map should assign that label to every point."""
        label_2d = np.full((IMG_H, IMG_W), 7, dtype=np.int32)
        pts = np.random.randn(30, 3)
        labels_3d = build_panorama_label_map(label_2d, pts, IMG_H, IMG_W)
        assert (labels_3d == 7).all()


class TestRenderLabelPanorama:
    """Test rendering labeled 3D points back to a 2D panorama."""

    def test_output_shape(self):
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        labels = np.array([1, 2])
        result = render_label_panorama(pts, labels, IMG_H, IMG_W)
        assert result.shape == (IMG_H, IMG_W)
        assert result.dtype == np.int32

    def test_empty_points_give_zeros(self):
        pts = np.zeros((0, 3))
        labels = np.zeros(0, dtype=np.int32)
        result = render_label_panorama(pts, labels, IMG_H, IMG_W)
        assert (result == 0).all()
