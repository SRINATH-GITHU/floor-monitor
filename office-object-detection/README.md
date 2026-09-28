# Office Object Detection

An offline research starter project for investigating small-object detection in software/IT office CCTV footage. Classes in `data/classes.txt` map to IDs: 0 pen, 1 bag, 2 book, 3 phone, 4 bottle.

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

## Local annotation

Run the OpenCV annotation app from this project directory after extracting frames:

```powershell
python scripts/annotate_frames.py
```

It opens images from `data/extracted_frames/` (including subfolders) and saves original image files and YOLO labels under `data/diagnostic/images/` and `data/diagnostic/labels/`. Display scaling is only for the window; saved images retain their original resolution. Everything runs locally and the scripts make no network requests.

| Key | Action |
| --- | --- |
| `0`–`4` | Select pen, bag, book, phone, or bottle |
| Left mouse drag | Draw a box |
| `s` | Save current image and annotations |
| `n` / `p` | Save and move to next / previous image |
| `u` | Undo most recent box |
| `c` | Clear all boxes for the current image |
| `q` or `Esc` | Quit |

The app loads an existing label for the current image when available. Navigation saves automatically; `c` takes effect when the cleared annotation is saved or when navigating. The image filename and selected class are shown in the window title.

Analyze saved annotations and box sizes with:

```powershell
python scripts/analyze_annotations.py
```

The report is printed per class and includes pixel width/height statistics, percentiles, area, area relative to a 1920×1080 frame, and width-bin counts. Use `--images-dir`, `--labels-dir`, `--frame-width`, and `--frame-height` to analyze another local dataset or reference frame size.

The analysis reports mean grayscale brightness, Laplacian variance as a basic sharpness indicator, and mean grayscale difference between sampled frames as a basic variation indicator. Values are useful for exploration only; they are not definitive object-detection metrics and do not measure whether target objects can be detected.

## Project layout

```text
data/       Local videos, sampled frames, diagnostic dataset, class names
scripts/    Metadata, frame extraction, annotation, and analysis
training/   Reserved for future work
inference/  Reserved for future work
models/     Reserved for manually managed model artifacts
results/    Reserved for analysis outputs
```
