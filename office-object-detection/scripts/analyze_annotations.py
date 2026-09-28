"""Summarize YOLO annotation box sizes using only local dataset files."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

import cv2


CLASSES = {0: "pen", 1: "bag", 2: "book", 3: "phone", 4: "bottle"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WIDTH_BINS = [(0, 10, "<10 px"), (10, 20, "10–20 px"), (20, 40, "20–40 px"), (40, 80, "40–80 px"), (80, 160, "80–160 px"), (160, math.inf, ">160 px")]


def describe(values: list[float]) -> str:
    if not values:
        return "count=0"
    ordered = sorted(values)
    return (f"count={len(values)}, min={min(values):.2f}, max={max(values):.2f}, mean={statistics.mean(values):.2f}, "
            f"median={statistics.median(values):.2f}, p10={statistics.quantiles(ordered, n=10, method='inclusive')[0]:.2f}, "
            f"p25={statistics.quantiles(ordered, n=4, method='inclusive')[0]:.2f}, p75={statistics.quantiles(ordered, n=4, method='inclusive')[2]:.2f}, "
            f"p90={statistics.quantiles(ordered, n=10, method='inclusive')[8]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("data/diagnostic/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/diagnostic/labels"))
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    args = parser.parse_args()
    images = {p.stem: p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS} if args.images_dir.exists() else {}
    by_class: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"width": [], "height": [], "area": [], "normalized_area": []})
    counts: dict[int, list[int]] = defaultdict(lambda: [0] * len(WIDTH_BINS))
    for label in sorted(args.labels_dir.glob("*.txt")) if args.labels_dir.exists() else []:
        image_path = images.get(label.stem)
        if image_path is None:
            print(f"WARNING: no matching image for {label.name}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"WARNING: could not read {image_path}")
            continue
        height, width = image.shape[:2]
        for line_no, row in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            parts = row.split()
            try:
                if len(parts) != 5:
                    raise ValueError("expected five values")
                cls = int(parts[0])
                cx, cy, nw, nh = map(float, parts[1:])
                if cls not in CLASSES or not all(math.isfinite(v) for v in (cx, cy, nw, nh)) or nw < 0 or nh < 0:
                    raise ValueError("invalid class or box values")
            except ValueError as exc:
                print(f"WARNING: {label.name}:{line_no}: {exc}; skipped")
                continue
            bw, bh = nw * width, nh * height
            area = bw * bh
            data = by_class[cls]
            data["width"].append(bw)
            data["height"].append(bh)
            data["area"].append(area)
            data["normalized_area"].append(area / (args.frame_width * args.frame_height))
            for i, (lo, hi, _) in enumerate(WIDTH_BINS):
                if lo <= bw < hi:
                    counts[cls][i] += 1
                    break
    for cls, name in CLASSES.items():
        data = by_class[cls]
        print(f"\n{name} (class {cls}) — {len(data['area'])} objects")
        print(f"  width px:  {describe(data['width'])}")
        print(f"  height px: {describe(data['height'])}")
        print(f"  area px²:  {describe(data['area'])}")
        print(f"  area / {args.frame_width}x{args.frame_height} frame: {describe(data['normalized_area'])}")
        print("  width ranges: " + ", ".join(f"{name}={count}" for (_, _, name), count in zip(WIDTH_BINS, counts[cls])))


if __name__ == "__main__":
    main()
