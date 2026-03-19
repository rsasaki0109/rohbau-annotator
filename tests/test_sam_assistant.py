"""Tests for SAM2 assistant integration."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _import_sam_assistant():
    """Import (or re-import) sam_assistant module."""
    import rohbau_annotator.sam_assistant as mod

    return mod


class TestIsSam2Available:
    """Test the availability check function."""

    def test_returns_bool(self):
        mod = _import_sam_assistant()
        result = mod.is_sam2_available()
        assert isinstance(result, bool)


class TestSAMAssistantWithoutSAM2:
    """Test SAMAssistant behavior when SAM2 is not installed."""

    def test_raises_runtime_error(self):
        mod = _import_sam_assistant()
        with patch.object(mod, "_SAM2_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="SAM2 is not installed"):
                mod.SAMAssistant()


class TestSAMAssistantWithMock:
    """Test SAMAssistant with mocked SAM2 backend."""

    IMG_H, IMG_W = 256, 512

    @pytest.fixture()
    def assistant(self):
        """Create a SAMAssistant with fully mocked SAM2 internals."""
        mock_predictor = MagicMock()

        # Configure predict to return plausible shapes
        num_masks = 3
        masks = np.zeros((num_masks, self.IMG_H, self.IMG_W), dtype=bool)
        masks[0, 50:150, 100:300] = True  # best mask
        scores = np.array([0.95, 0.80, 0.60])
        logits = np.random.randn(num_masks, self.IMG_H, self.IMG_W).astype(
            np.float32
        )
        mock_predictor.predict.return_value = (masks, scores, logits)

        mock_build_sam2 = MagicMock(return_value=MagicMock())
        mock_pred_cls = MagicMock(return_value=mock_predictor)

        mod = _import_sam_assistant()
        with (
            patch.object(mod, "_SAM2_AVAILABLE", True),
            patch.object(mod, "build_sam2", mock_build_sam2, create=True),
            patch.object(mod, "SAM2ImagePredictor", mock_pred_cls, create=True),
            patch.dict("sys.modules", {"torch": MagicMock()}),
        ):
            sam = mod.SAMAssistant(device="cpu")
            yield sam

    def test_set_image(self, assistant):
        image = np.random.randint(0, 255, (self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        assistant.set_image(image)
        assert assistant._image_set is True

    def test_predict_from_points_shape(self, assistant):
        image = np.random.randint(0, 255, (self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        assistant.set_image(image)

        points = np.array([[200, 100]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        masks, scores, logits = assistant.predict_from_points(points, labels)

        assert masks.shape[0] == 3
        assert masks.shape[1] == self.IMG_H
        assert masks.shape[2] == self.IMG_W
        assert scores.shape == (3,)

    def test_predict_from_box_shape(self, assistant):
        image = np.random.randint(0, 255, (self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        assistant.set_image(image)

        box = np.array([100, 50, 300, 150], dtype=np.float32)
        masks, scores, logits = assistant.predict_from_box(box)

        assert masks.shape[0] == 3
        assert masks.shape[1] == self.IMG_H
        assert masks.shape[2] == self.IMG_W

    def test_predict_best_mask_shape(self, assistant):
        image = np.random.randint(0, 255, (self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        assistant.set_image(image)

        points = np.array([[200, 100]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        mask = assistant.predict_best_mask(points=points, labels=labels)

        assert mask.shape == (self.IMG_H, self.IMG_W)
        assert mask.dtype == bool

    def test_predict_without_set_image_raises(self, assistant):
        points = np.array([[200, 100]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        with pytest.raises(RuntimeError, match="set_image"):
            assistant.predict_from_points(points, labels)

    def test_predict_best_mask_selects_highest_score(self, assistant):
        image = np.random.randint(0, 255, (self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        assistant.set_image(image)

        points = np.array([[200, 100]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        mask = assistant.predict_best_mask(points=points, labels=labels)

        # The mock returns masks[0] with score 0.95 as best
        # masks[0] has True in region [50:150, 100:300]
        assert mask[100, 200] is np.True_


class TestAnnotatorSAMIntegration:
    """Test that PanoramaAnnotator integrates SAM2 mode correctly."""

    def test_annotator_accepts_sam_assistant(self):
        from rohbau_annotator.annotator import PanoramaAnnotator

        panorama = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        # Pass None for sam_assistant -- should work fine
        annotator = PanoramaAnnotator(panorama=panorama, sam_assistant=None)
        assert annotator._sam_mode is False

    def test_annotator_sam_mode_toggle(self):
        from rohbau_annotator.annotator import PanoramaAnnotator
        from rohbau_annotator.sam_assistant import SAMAssistant

        panorama = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_sam = MagicMock(spec=SAMAssistant)
        annotator = PanoramaAnnotator(panorama=panorama, sam_assistant=mock_sam)

        assert annotator._sam_mode is False
        annotator._sam_mode = True
        assert annotator._sam_mode is True
