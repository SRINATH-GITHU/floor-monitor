# Office Object Detection

An offline research starter project for investigating small-object detection in software/IT office CCTV footage. Initial classes are listed in `data/classes.txt`: pen, paper, book, phone, and bottle.

All scripts operate on local files using OpenCV. They do not upload footage, access CCTV services, or make network requests. Keep confidential recordings in `data/raw_videos/` and restrict access according to your organization's policies. No YOLO training, model downloads, or inference pipeline is included yet.

## Setup

From this project directory, create a virtual environment and install the small dependency set:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Commands

Place a video in `data/raw_videos/`, then run from the project directory:

```powershell
python scripts/video_info.py "data/raw_videos/office_sample.mp4"
python scripts/analyze_video.py "data/raw_videos/office_sample.mp4" --interval 5
python scripts/extract_frames.py "data/raw_videos/office_sample.mp4" --interval 5
```

`--interval` is the number of seconds between samples. It defaults to 5 seconds so frame extraction does not produce a redundant sequence of consecutive images. Extracted frames keep the source resolution and are written under `data/extracted_frames/`. Use `--output-dir` to choose another local directory.

The analysis reports mean grayscale brightness, Laplacian variance as a basic sharpness indicator, and mean grayscale difference between sampled frames as a basic variation indicator. Values are useful for exploration only; they are not definitive object-detection metrics and do not measure whether target objects can be detected.

## Project layout

```text
data/       Local videos, sampled frames, dataset workspace, class names
scripts/    Metadata, frame extraction, and quality-indicator analysis
training/   Reserved for future work
inference/  Reserved for future work
models/     Reserved for manually managed model artifacts
results/    Reserved for analysis outputs
```
