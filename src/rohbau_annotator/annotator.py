"""Matplotlib-based panorama annotation tool.

Displays a panorama image and lets the user paint regions with semantic
class labels using a brush tool.  Labels are stored as a 2D label map
and can be back-projected to 3D via the projection module.

Supports an optional SAM2 mode for semi-automatic segmentation:
when enabled, clicks become SAM2 prompts instead of brush strokes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from rohbau_annotator.classes import CLASSES, CLASS_BY_ID, class_color_norm
from rohbau_annotator.projection import build_panorama_label_map
from rohbau_annotator.sam_assistant import SAMAssistant, is_sam2_available


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
    sam_assistant : SAMAssistant, optional
        Pre-initialized SAM2 assistant for semi-automatic annotation.
    """

    def __init__(
        self,
        panorama: np.ndarray,
        label_map: Optional[np.ndarray] = None,
        brush_radius: int = 10,
        sam_assistant: Optional[SAMAssistant] = None,
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

        # SAM2 mode state
        self.sam_assistant = sam_assistant
        self._sam_mode = False
        self._sam_fg_points: list[tuple[int, int]] = []
        self._sam_bg_points: list[tuple[int, int]] = []
        self._sam_mask: Optional[np.ndarray] = None  # (H, W) bool
        self._sam_overlay_artist = None
        self._sam_point_artists: list = []

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
    # SAM2 helpers
    # ------------------------------------------------------------------

    def _sam_update_prediction(self) -> None:
        """Run SAM2 prediction with current points and display result."""
        if self.sam_assistant is None:
            return

        all_points = self._sam_fg_points + self._sam_bg_points
        if not all_points:
            self._sam_clear_preview()
            return

        points = np.array(all_points, dtype=np.float32)
        labels = np.array(
            [1] * len(self._sam_fg_points) + [0] * len(self._sam_bg_points),
            dtype=np.int32,
        )

        self._sam_mask = self.sam_assistant.predict_best_mask(
            points=points, labels=labels
        )
        self._sam_show_preview()

    def _sam_show_preview(self) -> None:
        """Display the SAM2 proposed mask as a translucent overlay."""
        if self._sam_mask is None:
            return

        cls = CLASS_BY_ID.get(self.current_class_id, CLASSES[1])
        r, g, b = class_color_norm(cls)

        preview = np.zeros((self.img_h, self.img_w, 4), dtype=np.float32)
        preview[self._sam_mask] = [r, g, b, 0.5]

        if self._sam_overlay_artist is None:
            self._sam_overlay_artist = self.ax_img.imshow(preview)
        else:
            self._sam_overlay_artist.set_data(preview)

        self.fig.canvas.draw_idle()

    def _sam_clear_preview(self) -> None:
        """Remove SAM2 preview overlay."""
        if self._sam_overlay_artist is not None:
            blank = np.zeros((self.img_h, self.img_w, 4), dtype=np.float32)
            self._sam_overlay_artist.set_data(blank)

        for artist in self._sam_point_artists:
            artist.remove()
        self._sam_point_artists.clear()
        self.fig.canvas.draw_idle()

    def _sam_draw_points(self) -> None:
        """Draw foreground (green) and background (red) prompt points."""
        for artist in self._sam_point_artists:
            artist.remove()
        self._sam_point_artists.clear()

        for x, y in self._sam_fg_points:
            (pt,) = self.ax_img.plot(x, y, "g+", markersize=12, markeredgewidth=2)
            self._sam_point_artists.append(pt)
        for x, y in self._sam_bg_points:
            (pt,) = self.ax_img.plot(x, y, "rx", markersize=12, markeredgewidth=2)
            self._sam_point_artists.append(pt)
        self.fig.canvas.draw_idle()

    def _sam_accept_mask(self) -> None:
        """Accept the current SAM2 mask: write it to the label map."""
        if self._sam_mask is not None:
            self.label_map[self._sam_mask] = self.current_class_id
        self._sam_reset()
        self._refresh_overlay()

    def _sam_reject_mask(self) -> None:
        """Reject the current SAM2 mask and clear prompts."""
        self._sam_reset()

    def _sam_reset(self) -> None:
        """Reset SAM2 interaction state."""
        self._sam_fg_points.clear()
        self._sam_bg_points.clear()
        self._sam_mask = None
        self._sam_clear_preview()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax_img:
            return

        if self._sam_mode and self.sam_assistant is not None:
            cx = int(round(event.xdata))
            cy = int(round(event.ydata))
            if event.button == 1:  # left click = foreground
                self._sam_fg_points.append((cx, cy))
            elif event.button == 3:  # right click = background
                self._sam_bg_points.append((cx, cy))
            else:
                return
            self._sam_draw_points()
            self._sam_update_prediction()
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

    def _on_key(self, event) -> None:
        """Handle keyboard events for SAM2 accept/reject."""
        if not self._sam_mode:
            return
        if event.key == "enter":
            self._sam_accept_mask()
        elif event.key == "escape":
            self._sam_reject_mask()

    def _on_class_select(self, label: str) -> None:
        for cls in CLASSES:
            if cls.name == label:
                self.current_class_id = cls.id
                break

    def _on_brush_change(self, val: float) -> None:
        self.brush_radius = int(round(val))

    def _on_clear(self, event) -> None:
        self.label_map[:] = 0
        self._sam_reset()
        self._refresh_overlay()

    def _on_sam_toggle(self, event) -> None:
        """Toggle SAM2 mode on/off."""
        self._sam_mode = not self._sam_mode
        if self._sam_mode:
            self.btn_sam.label.set_text("SAM2: ON")
            self.ax_img.set_title(
                "SAM2 mode: L-click=foreground, R-click=background | "
                "Enter=accept, Esc=reject"
            )
        else:
            self.btn_sam.label.set_text("SAM2: OFF")
            self.ax_img.set_title("Paint with left mouse button. Close window to finish.")
            self._sam_reset()
        self.fig.canvas.draw_idle()

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
        ax_clear = self.fig.add_axes([0.65, 0.02, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, "Clear")
        self.btn_clear.on_clicked(self._on_clear)

        # SAM2 toggle button
        sam_label = "SAM2: OFF"
        if self.sam_assistant is None:
            sam_label = "SAM2: N/A"
        ax_sam = self.fig.add_axes([0.77, 0.02, 0.1, 0.05])
        self.btn_sam = Button(ax_sam, sam_label)
        if self.sam_assistant is not None:
            self.btn_sam.on_clicked(self._on_sam_toggle)

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # If SAM2 is available, set the image
        if self.sam_assistant is not None:
            self.sam_assistant.set_image(self.panorama)

        plt.show()
        return self.label_map


def annotate_scan(
    panorama: np.ndarray,
    points: np.ndarray,
    existing_labels: Optional[np.ndarray] = None,
    brush_radius: int = 10,
    sam_assistant: Optional[SAMAssistant] = None,
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
    sam_assistant : SAMAssistant, optional
        Pre-initialized SAM2 assistant for semi-automatic annotation.

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
        sam_assistant=sam_assistant,
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
