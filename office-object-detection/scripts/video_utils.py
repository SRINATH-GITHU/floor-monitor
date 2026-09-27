"""Shared local video utilities for the analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoError(Exception):
    """Raised when a video cannot be opened or read reliably."""


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, dict[str, object]]:
    video_path = Path(path).expanduser()
    if not video_path.exists() or not video_path.is_file():
        raise VideoError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"Unsupported or unreadable video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise VideoError(f"Video has invalid or unavailable stream metadata: {video_path}")

    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    metadata: dict[str, object] = {
        "path": video_path,
        "filename": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps,
        "codec": codec or None,
    }
    return capture, metadata


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sample_frames(
    capture: cv2.VideoCapture, fps: float, interval_seconds: float
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frames at fixed time intervals using seeks to avoid redundant reads."""
    if interval_seconds <= 0:
        raise ValueError("Sampling interval must be greater than zero seconds.")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * interval_seconds)))
    for frame_index in range(0, frame_count, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            # Corrupt or truncated videos may fail before their advertised end.
            if frame_index == 0:
                raise VideoError("Could not decode the first video frame; the file may be corrupt.")
            break
        yield frame_index, frame_index / fps, frame


def print_metadata(metadata: dict[str, object]) -> None:
    print(f"Filename: {metadata['filename']}")
    print(f"Resolution: {metadata['width']} x {metadata['height']}")
    print(f"FPS: {metadata['fps']:.3f}")
    print(f"Frame count: {metadata['frame_count']}")
    print(f"Duration: {float(metadata['duration']):.2f} seconds ({format_duration(float(metadata['duration']))})")
    print(f"Codec: {metadata['codec'] or 'unavailable'}")
