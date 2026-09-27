"""Extract sparsely sampled full-resolution frames from a local video."""

import argparse
from pathlib import Path

import cv2

from video_utils import VideoError, open_video, sample_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a local video file")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between frames (default: 5)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "extracted_frames")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    capture = None
    try:
        capture, metadata = open_video(args.video)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(metadata["filename"])).stem
        saved = 0
        for frame_index, timestamp, frame in sample_frames(capture, float(metadata["fps"]), args.interval):
            output = args.output_dir / f"{stem}_t{timestamp:010.2f}s_f{frame_index:08d}.jpg"
            if not cv2.imwrite(str(output), frame):
                raise VideoError(f"Could not write sampled frame: {output}")
            saved += 1
        if saved == 0:
            raise VideoError("No frames could be decoded from the video.")
        print(f"Saved {saved} frame(s) at original resolution to: {args.output_dir.resolve()}")
        print("Frames were processed locally; no network access is performed.")
    except (VideoError, OSError) as exc:
        parser.error(str(exc))
    finally:
        if capture is not None:
            capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
