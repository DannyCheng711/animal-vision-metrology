# COCO Dataset Setup

## Dataset Selection

Use COCO 2017 validation images and the original instance annotations for baseline inference. Keep the annotations unchanged for later segmentation evaluation.

## Download

Run from the project root. These HTTPS URLs address the official COCO S3 bucket directly.

```bash
mkdir -p data/coco
# Download COCO 2017 validation images
curl -fL --retry 3 -C - \
  -o data/coco/val2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip

# Download COCO 2017 annotations
curl -fL --retry 3 -C - \
  -o data/coco/annotations_trainval2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract validation images
unzip -oq data/coco/val2017.zip -d data/coco

# Extract only validation instance annotations
unzip -o data/coco/annotations_trainval2017.zip \
  annotations/instances_val2017.json -d data/coco
```


## Project Structure

```text
computer-vision/
├── dataset/
│   ├── dataloader.py
│   └── prepro.py
├── model/
│   ├── __init__.py
│   ├── animal_segmentation.py
│   ├── eye_localization.py
│   └── eye_segmentation.py
├── measurement.py
├── run/
│   ├── inference.py
│   └── evaluation.py
├── docs/
├── data/
├── outputs/
└── weights/
```

## Module Responsibilities

- `dataset/dataloader.py`: read images and original COCO annotations without changing coordinates.
- `dataset/prepro.py`: filter animals, export selection CSV, and prepare crops with coordinate transformation metadata.
- `model/animal_segmentation.py`: animal instance segmentation model wrapper.
- `model/eye_localization.py`: eye localization model wrapper; return eye coordinates and confidence scores in original-image coordinates.
- `model/eye_segmentation.py`: eye segmentation model wrapper; return masks and quality scores aligned with the original image.
- `measurement.py`: calculate pixel distances from coordinates or masks; no model loading or dataset reading.
- `run/inference.py`: orchestrate dataset reading, model inference, measurement, and result export.
- `run/evaluation.py`: compare predictions with reference annotations and aggregate error metrics when reference annotations are available.
- `weights/`: local model configurations and checkpoints, excluded from Git.
- `data/`: local datasets, excluded from Git.
- `outputs/`: generated model predictions, masks, measurements, evaluation reports, and dataset selection files; excluded from Git.

`measurement.py`, `run/inference.py`, and `run/evaluation.py` are initially empty placeholders. Existing model and dataset implementations will be integrated into these responsibilities separately.

COCO bounding boxes use `[x, y, width, height]` in original image pixels. Preserve instance annotation IDs and original image dimensions throughout preprocessing. Convert crop or resized coordinates back to original-image coordinates before measurement. Distances are measured in pixels unless a separate scale calibration is available.
