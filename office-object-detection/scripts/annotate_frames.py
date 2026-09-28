"""Local mouse-driven YOLO box annotation for extracted CCTV frames."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


CLASSES = {0: "pen", 1: "bag", 2: "book", 3: "phone", 4: "bottle"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class Annotator:
    def __init__(self, image_paths: list[Path], output_dir: Path, window: str) -> None:
        self.paths = image_paths
        self.output = output_dir
        self.images_out = output_dir / "images"
        self.labels_out = output_dir / "labels"
        self.images_out.mkdir(parents=True, exist_ok=True)
        self.labels_out.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.index = 0
        self.class_id = 0
        self.boxes: list[tuple[int, int, int, int, int]] = []
        self.image = None
        self.display_scale = 1.0
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None

    def load(self) -> bool:
        self.image = cv2.imread(str(self.paths[self.index]), cv2.IMREAD_COLOR)
        if self.image is None:
            print(f"Could not read image: {self.paths[self.index]}")
            return False
        label_path = self.labels_out / f"{self.paths[self.index].stem}.txt"
        self.boxes = []
        if label_path.exists():
            height, width = self.image.shape[:2]
            for row in label_path.read_text(encoding="utf-8").splitlines():
                parts = row.split()
                if len(parts) != 5:
                    continue
                try:
                    cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                except ValueError:
                    continue
                x1, y1 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
                x2, y2 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
                self.boxes.append((cls, x1, y1, x2, y2))
        return True

    def mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            a, b = self.drag_start, (x, y)
            x1, x2 = sorted((round(a[0] / self.display_scale), round(b[0] / self.display_scale)))
            y1, y2 = sorted((round(a[1] / self.display_scale), round(b[1] / self.display_scale)))
            h, w = self.image.shape[:2]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                self.boxes.append((self.class_id, x1, y1, x2, y2))
            self.drag_start = self.drag_end = None

    def render(self) -> None:
        h, w = self.image.shape[:2]
        max_w, max_h = 1280, 760
        self.display_scale = min(max_w / w, max_h / h, 1.0)
        view = cv2.resize(self.image, (max(1, round(w * self.display_scale)), max(1, round(h * self.display_scale)))) if self.display_scale != 1 else self.image.copy()
        for cls, x1, y1, x2, y2 in self.boxes:
            p1 = (round(x1 * self.display_scale), round(y1 * self.display_scale))
            p2 = (round(x2 * self.display_scale), round(y2 * self.display_scale))
            cv2.rectangle(view, p1, p2, (60, 220, 60), 2)
            cv2.putText(view, CLASSES.get(cls, str(cls)), (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2)
        if self.drag_start and self.drag_end:
            cv2.rectangle(view, self.drag_start, self.drag_end, (0, 220, 255), 1)
        title = f"{self.paths[self.index].name} ({self.index + 1}/{len(self.paths)}) | class {self.class_id}: {CLASSES[self.class_id]} | boxes {len(self.boxes)}"
        cv2.imshow(self.window, view)
        cv2.setWindowTitle(self.window, title)

    def save(self) -> None:
        path = self.paths[self.index]
        target_image = self.images_out / path.name
        if path.resolve() != target_image.resolve():
            shutil.copy2(path, target_image)
        h, w = self.image.shape[:2]
        rows = []
        for cls, x1, y1, x2, y2 in self.boxes:
            cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            rows.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        (self.labels_out / f"{path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"Saved {target_image.name}: {len(rows)} box(es)")

    def run(self) -> None:
        if not self.load():
            return
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse)
        while True:
            self.render()
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            elif ord("0") <= key <= ord("4"):
                self.class_id = key - ord("0")
            elif key in (ord("s"),):
                self.save()
            elif key in (ord("u"), 8, 127):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord("c"):
                self.boxes.clear()
            elif key in (ord("n"), 83):
                self.save()
                self.index = (self.index + 1) % len(self.paths)
                self.load()
            elif key in (ord("p"), 81):
                self.save()
                self.index = (self.index - 1) % len(self.paths)
                self.load()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/extracted_frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostic"))
    args = parser.parse_args()
    paths = sorted(p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        parser.error(f"No supported images found under {args.input_dir}")
    Annotator(paths, args.output_dir, "Local CCTV YOLO Annotator").run()


if __name__ == "__main__":
    main()
