"""Summarize video metadata and sampled-frame quality indicators."""

import argparse

import cv2
import numpy as np

from video_utils import VideoError, open_video, print_metadata, sample_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a local video file")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between analysis samples (default: 5)")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    capture = None
    try:
        capture, metadata = open_video(args.video)
        print_metadata(metadata)

        brightness_values: list[float] = []
        sharpness_values: list[float] = []
        variation_values: list[float] = []
        previous_gray = None
        sampled = 0
        for _, _, frame in sample_frames(capture, float(metadata["fps"]), args.interval):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(np.mean(gray)))
            sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if previous_gray is not None:
                variation_values.append(float(np.mean(cv2.absdiff(gray, previous_gray))))
            previous_gray = gray
            sampled += 1
        if sampled == 0:
            raise VideoError("No frames could be decoded from the video.")

        print(f"Sample interval: {args.interval:g} seconds")
        print(f"Frames analyzed: {sampled}")
        print(f"Mean brightness (0-255): {np.mean(brightness_values):.2f}")
        print(f"Mean sharpness (Laplacian variance): {np.mean(sharpness_values):.2f}")
        if variation_values:
            print(f"Mean sampled-frame variation (mean absolute grayscale difference, 0-255): {np.mean(variation_values):.2f}")
        else:
            print("Mean sampled-frame variation: unavailable (only one frame sampled)")
        print("These are basic image-quality indicators, not definitive object-detection metrics.")
        print("All video processing is local; this script does not access a network service.")
    except (VideoError, OSError) as exc:
        parser.error(str(exc))
    finally:
        if capture is not None:
            capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
