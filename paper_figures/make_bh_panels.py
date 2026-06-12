"""Split the current paper FIG2/FIG3 black-hole panels.

The tracked paper figures already contain the correct calculation results:
  paper/figure/p=0_BH.png -> FIG2, p=0
  paper/figure/p=1_BH.png -> FIG3, p=1

This script separates those four-panel PNGs into per-panel files:
  paper_figures/generated/FIG2/p=0_BH_a.png ... p=0_BH_d.png
  paper_figures/generated/FIG3/p=1_BH_a.png ... p=1_BH_d.png

Do not rebuild these panels from paper_data/p{0,1}/lr_p and ur_p_2 unless the
matching panel-specific runs are available. Those archived CSVs do not contain
the latest FIG2/FIG3 initial conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .common import PAPER_FIG_ORIG, ensure_fig_dir
except ImportError:
    from common import PAPER_FIG_ORIG, ensure_fig_dir


FIGURES = [
    ("FIG2", PAPER_FIG_ORIG / "p=0_BH.png", "p=0_BH"),
    ("FIG3", PAPER_FIG_ORIG / "p=1_BH.png", "p=1_BH"),
]
PANEL_TAGS = ("a", "b", "c", "d")


def _load_font():
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def _transparent_runs(mask, *, min_len=6):
    runs = []
    in_run = False
    for idx, is_transparent in enumerate(mask):
        if is_transparent and not in_run:
            start = idx
            in_run = True
        if in_run and (not is_transparent or idx == len(mask) - 1):
            end = idx - 1 if not is_transparent else idx
            if end - start + 1 >= min_len:
                runs.append((start, end))
            in_run = False
    return runs


def _panel_x_ranges(alpha):
    transparent_fraction = (alpha < 10).mean(axis=0)
    transparent_runs = _transparent_runs(transparent_fraction > 0.85)
    if len(transparent_runs) >= 4:
        separators = transparent_runs[:4]
        width = alpha.shape[1]
        return [
            (separators[0][1] + 1, separators[1][0]),
            (separators[1][1] + 1, separators[2][0]),
            (separators[2][1] + 1, separators[3][0]),
            (separators[3][1] + 1, width),
        ]

    width = alpha.shape[1]
    step = width // 4
    return [(i * step, (i + 1) * step if i < 3 else width) for i in range(4)]


def _content_y0(alpha):
    transparent_fraction = (alpha < 10).mean(axis=1)
    rows = np.where(transparent_fraction < 0.85)[0]
    return int(rows[0]) if len(rows) else 0


def split_figure(fig_name: str, src: Path, prefix: str):
    if not src.exists():
        raise FileNotFoundError(f"missing source figure: {src}")

    rgba = Image.open(src).convert("RGBA")
    alpha = np.array(rgba)[:, :, 3]
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white, rgba).convert("RGB")

    out_dir = ensure_fig_dir() / fig_name
    out_dir.mkdir(parents=True, exist_ok=True)

    font = _load_font()
    top_margin = 28
    y0 = _content_y0(alpha)
    x_ranges = _panel_x_ranges(alpha)

    for tag, (x0, x1) in zip(PANEL_TAGS, x_ranges):
        panel = composite.crop((x0, y0, x1, rgba.height))
        canvas = Image.new("RGB", (panel.width, panel.height + top_margin), "white")
        canvas.paste(panel, (0, top_margin))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), f"({tag})", fill="black", font=font)

        out = out_dir / f"{prefix}_{tag}.png"
        canvas.save(out)
        print(f"[bh] saved -> {out}")


def main():
    for fig_name, src, prefix in FIGURES:
        split_figure(fig_name, src, prefix)


if __name__ == "__main__":
    main()
