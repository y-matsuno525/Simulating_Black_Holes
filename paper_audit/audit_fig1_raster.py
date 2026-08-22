"""Digitize the retained Fig. 1 raster and compare it with Eq. (16)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "paper" / "figure" / "dispersion.png"
REPORT = REPO / "paper_audit" / "fig1_raster_audit.json"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    red, green, blue = np.moveaxis(image.astype(int), -1, 0)
    return {
        "lower": (blue > red + 45) & (blue > green + 20) & (green > red + 20),
        "upper": (red > green + 60) & (green > blue + 35),
    }


def digitized_curve(mask: np.ndarray, x_range: tuple[int, int], y_range: tuple[int, int]):
    xs = []
    ys = []
    for x in range(x_range[0], x_range[1] + 1):
        candidates = np.flatnonzero(mask[y_range[0] : y_range[1], x]) + y_range[0]
        if len(candidates):
            xs.append(x)
            ys.append(float(np.median(candidates)))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def main() -> None:
    image = np.asarray(Image.open(IMAGE).convert("RGB"))
    if image.shape[:2] != (1080, 1438):
        raise RuntimeError(f"unexpected retained Fig. 1 dimensions: {image.shape}")
    masks = color_masks(image)
    x_ranges = ((173, 477), (600, 897), (1016, 1317))
    rows = ((0, 0, (0, 250)), (1, 1, (250, 520)))
    report = {"image": str(IMAGE), "panels": {}}

    for row, p, y_range in rows:
        for column, beta in enumerate((0.0, 1.0, 2.0)):
            x_min, x_max = x_ranges[column]
            for branch, mask in masks.items():
                x_pixels, y_pixels = digitized_curve(mask, (x_min, x_max), y_range)
                k = -np.pi + 2 * np.pi * (x_pixels - x_min) / (x_max - x_min)
                root = np.sqrt(np.sin(k) ** 2 + p**2 * (1 - np.cos(k)) ** 2)
                energy = beta * np.sin(k) + (root if branch == "upper" else -root)
                design = np.column_stack([energy, np.ones_like(energy)])
                coefficients, *_ = np.linalg.lstsq(design, y_pixels, rcond=None)
                residual = y_pixels - design @ coefficients
                key = f"row{row}_{branch}_p{p}_beta{beta:g}"
                report["panels"][key] = {
                    "samples": int(len(x_pixels)),
                    "median_absolute_residual_pixels": float(np.median(np.abs(residual))),
                    "percentile95_absolute_residual_pixels": float(
                        np.quantile(np.abs(residual), 0.95)
                    ),
                }

    maximum_median = max(
        panel["median_absolute_residual_pixels"]
        for panel in report["panels"].values()
    )
    report["maximum_median_absolute_residual_pixels"] = maximum_median
    report["pass_threshold_pixels"] = 5.0
    report["passed"] = maximum_median < report["pass_threshold_pixels"]
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[fig1-raster-audit] max median residual = {maximum_median:.3f} px")
    print(f"[fig1-raster-audit] wrote {REPORT}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
