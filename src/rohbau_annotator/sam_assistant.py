"""SAM2 (Segment Anything Model 2) integration for semi-automatic annotation.

Wraps SAM2 to provide interactive segmentation on panorama images.
Users click foreground/background points and SAM2 proposes masks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# SAM2 availability flag
_SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _SAM2_AVAILABLE = True
except ImportError:
    pass


def is_sam2_available() -> bool:
    """Return True if the SAM2 package is installed."""
    return _SAM2_AVAILABLE


class SAMAssistant:
    """Wrapper around SAM2 for interactive segmentation.

    Parameters
    ----------
    model_cfg : str
        SAM2 model config name (e.g., ``"sam2_hiera_t"``).
    checkpoint : str
        Path to the SAM2 checkpoint file.
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_t",
        checkpoint: str = "sam2_hiera_tiny.pt",
        device: str = "cpu",
    ) -> None:
        if not _SAM2_AVAILABLE:
            raise RuntimeError(
                "SAM2 is not installed. Install it with:\n"
                "  pip install segment-anything-2\n"
                "or see https://github.com/facebookresearch/segment-anything-2"
            )

        import torch

        self.device = device
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self._image_set = False

    def set_image(self, image: np.ndarray) -> None:
        """Set the image for SAM2 prediction.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image.
        """
        self.predictor.set_image(image)
        self._image_set = True
        self._img_h, self._img_w = image.shape[:2]

    def predict_from_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks from point prompts.

        Parameters
        ----------
        points : np.ndarray
            (K, 2) array of (x, y) pixel coordinates.
        labels : np.ndarray
            (K,) array of labels: 1 = foreground, 0 = background.

        Returns
        -------
        masks : np.ndarray
            (M, H, W) boolean mask array. M is the number of candidate masks.
        scores : np.ndarray
            (M,) confidence scores for each mask.
        logits : np.ndarray
            (M, H, W) raw logit masks (can be used as input for refinement).
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before predict_from_points().")

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        return masks, scores, logits

    def predict_from_box(
        self,
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks from a bounding box prompt.

        Parameters
        ----------
        box : np.ndarray
            (4,) array of [x1, y1, x2, y2] in pixel coordinates.

        Returns
        -------
        masks : np.ndarray
            (M, H, W) boolean mask array.
        scores : np.ndarray
            (M,) confidence scores.
        logits : np.ndarray
            (M, H, W) raw logits.
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before predict_from_box().")

        masks, scores, logits = self.predictor.predict(
            box=box[None, :],  # SAM2 expects (B, 4)
            multimask_output=True,
        )
        return masks, scores, logits

    def predict_best_mask(
        self,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the single best mask from the given prompts.

        Parameters
        ----------
        points : np.ndarray, optional
            (K, 2) point coordinates.
        labels : np.ndarray, optional
            (K,) point labels (1=fg, 0=bg).
        box : np.ndarray, optional
            (4,) bounding box [x1, y1, x2, y2].

        Returns
        -------
        mask : np.ndarray
            (H, W) boolean mask for the best prediction.
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before predict_best_mask().")

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box[None, :] if box is not None else None,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        return masks[best_idx]
