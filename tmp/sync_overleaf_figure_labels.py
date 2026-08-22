from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROJECT = Path.cwd().parent / "black_hole_simulation (15)_A"
FIGURE_DIR = PROJECT / "figure"
BACKUP_DIR = FIGURE_DIR / "original_labels_before_sync"


def math_image(text: str, *, fontsize: float, dpi: int = 200) -> Image.Image:
    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    fig.patch.set_alpha(0)
    artist = fig.text(0, 0, text, fontsize=fontsize)
    fig.canvas.draw()
    bbox = artist.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.08, 1.15)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, transparent=True, bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()), pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def add_surface_gravity_labels() -> None:
    source_path = FIGURE_DIR / "sg.png"
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / source_path.name
    if not backup_path.exists():
        backup_path.write_bytes(source_path.read_bytes())

    source = Image.open(backup_path).convert("RGB")
    half = source.width // 2
    panels = [source.crop((0, 0, half, source.height)), source.crop((half, 0, source.width, source.height))]

    # Remove the old short ylabel while preserving every plotted data pixel.
    cropped = [panel.crop((42, 0, panel.width, panel.height)) for panel in panels]
    left_margin = 128
    top_margin = 42
    gap = 24
    panel_width = left_margin + max(panel.width for panel in cropped)
    canvas = Image.new("RGB", (2 * panel_width + gap, source.height + top_margin), "white")

    ylabel = math_image(r"$\delta\sigma_{\mathcal{E}}(t)$", fontsize=12)
    tags = [math_image(r"$(a)$", fontsize=11), math_image(r"$(b)$", fontsize=11)]
    for index, panel in enumerate(cropped):
        x0 = index * (panel_width + gap)
        canvas.paste(panel, (x0 + left_margin, top_margin))
        canvas.paste(
            ylabel,
            (x0 + 8, top_margin + (panel.height - ylabel.height) // 2),
            ylabel,
        )
        canvas.paste(tags[index], (x0 + left_margin - 18, 3), tags[index])

    canvas.save(source_path, dpi=(150, 150))


if __name__ == "__main__":
    add_surface_gravity_labels()
