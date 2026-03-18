"""Matplotlib-based panorama annotation tool.

Displays a panorama image and lets the user paint regions with semantic
class labels using a brush tool.  Labels are stored as a 2D label map
and can be back-projected to 3D via the projection module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from rohbau_annotator.classes import CLASSES, CLASS_BY_ID, class_color_norm
from rohbau_annotator.projection import build_panorama_label_map


class PanoramaAnnotator:
    """Interactive matplotlib annotation tool for panorama images.

    Parameters
    ----------
    panorama : np.ndarray
        (H, W, 3) uint8 panorama image to annotate.
    label_map : np.ndarray, optional
        (H, W) existing integer label map.  If None, starts blank.
    brush_radius : int
        Initial brush radius in pixels.
    """

    def __init__(
        self,
        panorama: np.ndarray,
        label_map: Optional[np.ndarray] = None,
        brush_radius: int = 10,
    ) -> None:
        self.panorama = panorama
        h, w = panorama.shape[:2]
        self.img_h = h
        self.img_w = w

        if label_map is not None:
            self.label_map = label_map.copy()
        else:
            self.label_map = np.zeros((h, w), dtype=np.int32)

        self.brush_radius = brush_radius
        self.current_class_id = 1  # default: wall
        self._painting = False

        # Build overlay
        self.overlay = self._build_overlay()

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _build_overlay(self) -> np.ndarray:
        """Build an RGBA overlay image from the label map."""
        overlay = np.zeros((self.img_h, self.img_w, 4), dtype=np.float32)
        for cls in CLASSES:
            if cls.id == 0:
                continue
            mask = self.label_map == cls.id
            if not mask.any():
                continue
            r, g, b = class_color_norm(cls)
            overlay[mask] = [r, g, b, 0.45]
        return overlay

    def _paint_circle(self, cx: int, cy: int) -> None:
        """Paint a filled circle on the label map."""
        rr = self.brush_radius
        yy, xx = np.ogrid[-rr : rr + 1, -rr : rr + 1]
        circle_mask = xx**2 + yy**2 <= rr**2

        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if not circle_mask[dy + rr, dx + rr]:
                    continue
                py = cy + dy
                px = cx + dx
                if 0 <= py < self.img_h and 0 <= px < self.img_w:
                    self.label_map[py, px] = self.current_class_id

    def _refresh_overlay(self) -> None:
        """Rebuild and redraw the overlay."""
        self.overlay = self._build_overlay()
        self._overlay_artist.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax_img:
            return
        self._painting = True
        self._on_motion(event)

    def _on_release(self, event) -> None:
        if self._painting:
            self._painting = False
            self._refresh_overlay()

    def _on_motion(self, event) -> None:
        if not self._painting or event.inaxes != self.ax_img:
            return
        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        self._paint_circle(cx, cy)

    def _on_class_select(self, label: str) -> None:
        for cls in CLASSES:
            if cls.name == label:
                self.current_class_id = cls.id
                break

    def _on_brush_change(self, val: float) -> None:
        self.brush_radius = int(round(val))

    def _on_clear(self, event) -> None:
        self.label_map[:] = 0
        self._refresh_overlay()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """Launch the annotation UI.  Returns the label map when the window is closed."""
        self.fig = plt.figure("Rohbau Annotator", figsize=(16, 8))

        # Main panorama axis
        self.ax_img = self.fig.add_axes([0.0, 0.1, 0.75, 0.85])
        self.ax_img.imshow(self.panorama)
        self._overlay_artist = self.ax_img.imshow(self.overlay)
        self.ax_img.set_axis_off()
        self.ax_img.set_title("Paint with left mouse button. Close window to finish.")

        # Class selector (radio buttons)
        ax_radio = self.fig.add_axes([0.77, 0.15, 0.22, 0.75])
        class_names = [c.name for c in CLASSES if c.id > 0]
        class_colors = [class_color_norm(c) for c in CLASSES if c.id > 0]
        self.radio = RadioButtons(ax_radio, class_names, activecolor="black")
        for lbl, color in zip(self.radio.labels, class_colors):
            lbl.set_color(color)
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_class_select)

        # Brush size slider
        ax_slider = self.fig.add_axes([0.1, 0.02, 0.5, 0.04])
        self.slider = Slider(ax_slider, "Brush", 1, 50, valinit=self.brush_radius, valstep=1)
        self.slider.on_changed(self._on_brush_change)

        # Clear button
        ax_clear = self.fig.add_axes([0.77, 0.02, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, "Clear")
        self.btn_clear.on_clicked(self._on_clear)

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        plt.show()
        return self.label_map


def annotate_scan(
    panorama: np.ndarray,
    points: np.ndarray,
    existing_labels: Optional[np.ndarray] = None,
    brush_radius: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Run annotation on a panorama and back-project labels to 3D.

    Parameters
    ----------
    panorama : np.ndarray
        (H, W, 3) panorama image.
    points : np.ndarray
        (N, 3) point cloud coordinates.
    existing_labels : np.ndarray, optional
        (H, W) existing 2D label map to continue editing.
    brush_radius : int
        Initial brush radius.

    Returns
    -------
    label_2d : np.ndarray
        (H, W) 2D label map on the panorama.
    label_3d : np.ndarray
        (N,) integer labels for each 3D point.
    """
    annotator = PanoramaAnnotator(
        panorama=panorama,
        label_map=existing_labels,
        brush_radius=brush_radius,
    )
    label_2d = annotator.run()

    img_h, img_w = panorama.shape[:2]
    label_3d = build_panorama_label_map(label_2d, points, img_h, img_w)

    return label_2d, label_3d


def save_label_map(label_map: np.ndarray, path: str | Path) -> None:
    """Save a 2D label map to a .npy file."""
    np.save(str(path), label_map)


def load_label_map(path: str | Path) -> np.ndarray:
    """Load a 2D label map from a .npy file."""
    return np.load(str(path))
