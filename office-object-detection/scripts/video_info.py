"""Print metadata for a local video without changing the source file."""

import argparse

from video_utils import VideoError, open_video, print_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a local video file")
    args = parser.parse_args()
    try:
        capture, metadata = open_video(args.video)
        capture.release()
        print_metadata(metadata)
    except (VideoError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
