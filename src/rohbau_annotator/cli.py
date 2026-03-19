"""CLI entry point for rohbau-annotator."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np


@click.group()
@click.version_option(package_name="rohbau-annotator")
def main() -> None:
    """Universal TLS Point Cloud Annotator with SAM2.

    Annotate Rohbau3D .npy directories, PLY, PCD, or LAS/LAZ point clouds.
    """


@main.command()
@click.argument("scan_path", type=click.Path(exists=True))
@click.option(
    "--label-map",
    type=click.Path(),
    default=None,
    help="Path to an existing 2D label map (.npy) to continue editing.",
)
@click.option("--brush-radius", type=int, default=10, help="Initial brush radius in pixels.")
@click.option("--sam2", is_flag=True, default=False, help="Enable SAM2 semi-automatic segmentation.")
@click.option("--sam2-checkpoint", type=click.Path(), default="sam2_hiera_tiny.pt", help="Path to SAM2 checkpoint file.")
@click.option("--sam2-model-cfg", type=str, default="sam2_hiera_t", help="SAM2 model config name.")
@click.option("--sam2-device", type=str, default="cpu", help="Device for SAM2 inference (cpu or cuda).")
def annotate(
    scan_path: str,
    label_map: str | None,
    brush_radius: int,
    sam2: bool,
    sam2_checkpoint: str,
    sam2_model_cfg: str,
    sam2_device: str,
) -> None:
    """Launch the panorama annotation UI for a scan directory or point cloud file.

    SCAN_PATH can be a Rohbau3D directory (containing .npy files) or a single
    PLY/PCD/LAS file.
    """
    from rohbau_annotator.annotator import annotate_scan, save_label_map
    from rohbau_annotator.loader import load_scan

    scan = load_scan(scan_path)
    if scan.img_color is None:
        raise click.ClickException(
            "No panorama image found. "
            "Rohbau3D directories require img_color.png; "
            "single-file point clouds do not include panorama images."
        )

    existing = None
    if label_map is not None:
        existing = np.load(label_map)
        click.echo(f"Loaded existing label map from {label_map}")

    # Initialize SAM2 assistant if requested
    sam_assistant = None
    if sam2:
        from rohbau_annotator.sam_assistant import SAMAssistant, is_sam2_available

        if not is_sam2_available():
            raise click.ClickException(
                "SAM2 is not installed. Install with: pip install -e '.[sam2]'"
            )
        click.echo(f"Loading SAM2 model ({sam2_model_cfg}) on {sam2_device}...")
        sam_assistant = SAMAssistant(
            model_cfg=sam2_model_cfg,
            checkpoint=sam2_checkpoint,
            device=sam2_device,
        )
        click.echo("SAM2 loaded. Use the SAM2 button in the UI to toggle mode.")

    click.echo(f"Scan: {scan.scan_dir.name}  |  {scan.num_points} points  |  panorama {scan.panorama_shape}")
    click.echo("Close the annotation window to save.")

    label_2d, label_3d = annotate_scan(
        panorama=scan.img_color,
        points=scan.coord,
        existing_labels=existing,
        brush_radius=brush_radius,
        sam_assistant=sam_assistant,
    )

    out_dir = Path(scan_path) if Path(scan_path).is_dir() else Path(scan_path).parent
    label_2d_path = out_dir / "label_2d.npy"
    label_3d_path = out_dir / "label_3d.npy"

    save_label_map(label_2d, label_2d_path)
    np.save(str(label_3d_path), label_3d)

    click.echo(f"Saved 2D label map: {label_2d_path}")
    click.echo(f"Saved 3D labels:    {label_3d_path}")


@main.command()
@click.argument("scan_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for 3D labels .npy.  Defaults to <scan_dir>/label_3d.npy.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["npy", "ply", "las"]),
    default="npy",
    help="Output format: npy (default), ply (with label field), or las (classification).",
)
def export(scan_path: str, output: str | None, output_format: str) -> None:
    """Export 2D panorama annotations to 3D point cloud labels.

    SCAN_PATH can be a Rohbau3D directory or a single PLY/PCD/LAS file.
    """
    from rohbau_annotator.loader import load_scan
    from rohbau_annotator.projection import build_panorama_label_map

    scan = load_scan(scan_path)
    scan_dir = Path(scan_path) if Path(scan_path).is_dir() else Path(scan_path).parent
    label_2d_path = scan_dir / "label_2d.npy"

    if not label_2d_path.exists():
        raise click.ClickException(f"No label_2d.npy found in {scan_dir}. Run 'annotate' first.")

    label_2d = np.load(str(label_2d_path))
    img_h, img_w = label_2d.shape[:2]

    click.echo(f"Back-projecting {img_h}x{img_w} label map to {scan.num_points} points...")
    label_3d = build_panorama_label_map(label_2d, scan.coord, img_h, img_w)

    if output_format == "npy":
        out_path = Path(output) if output else scan_dir / "label_3d.npy"
        np.save(str(out_path), label_3d)
        click.echo(f"Saved 3D labels: {out_path}")
    elif output_format == "ply":
        from rohbau_annotator.exporters import export_labeled_ply

        out_path = Path(output) if output else scan_dir / "labeled.ply"
        export_labeled_ply(scan.coord, label_3d, out_path, colors=scan.color)
        click.echo(f"Saved labeled PLY: {out_path}")
    elif output_format == "las":
        from rohbau_annotator.exporters import export_labeled_las

        out_path = Path(output) if output else scan_dir / "labeled.las"
        export_labeled_las(scan.coord, label_3d, out_path, colors=scan.color)
        click.echo(f"Saved labeled LAS: {out_path}")


@main.command()
@click.argument("scan_path", type=click.Path(exists=True))
def stats(scan_path: str) -> None:
    """Show annotation statistics for a scan directory or point cloud file."""
    from rohbau_annotator.classes import CLASS_BY_ID

    scan_dir = Path(scan_path) if Path(scan_path).is_dir() else Path(scan_path).parent
    label_2d_path = scan_dir / "label_2d.npy"
    label_3d_path = scan_dir / "label_3d.npy"

    click.echo(f"Scan: {scan_dir.name}")

    if label_2d_path.exists():
        label_2d = np.load(str(label_2d_path))
        total_px = label_2d.size
        labeled_px = np.count_nonzero(label_2d)
        click.echo(f"\n2D label map: {label_2d.shape[0]}x{label_2d.shape[1]}")
        click.echo(f"  Labeled pixels: {labeled_px}/{total_px} ({100*labeled_px/total_px:.1f}%)")
        _print_class_stats(label_2d)
    else:
        click.echo("\nNo label_2d.npy found.")

    if label_3d_path.exists():
        label_3d = np.load(str(label_3d_path))
        total_pts = label_3d.size
        labeled_pts = np.count_nonzero(label_3d)
        click.echo(f"\n3D labels: {total_pts} points")
        click.echo(f"  Labeled points: {labeled_pts}/{total_pts} ({100*labeled_pts/total_pts:.1f}%)")
        _print_class_stats(label_3d)
    else:
        click.echo("\nNo label_3d.npy found.")


def _print_class_stats(labels: np.ndarray) -> None:
    """Print per-class counts."""
    from rohbau_annotator.classes import CLASS_BY_ID

    unique, counts = np.unique(labels, return_counts=True)
    for class_id, count in zip(unique, counts):
        if class_id == 0:
            continue
        cls = CLASS_BY_ID.get(int(class_id))
        name = cls.name if cls else f"unknown({class_id})"
        click.echo(f"    {name:>12s}: {count:>10d}")


if __name__ == "__main__":
    main()
