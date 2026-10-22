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


## Module Responsibilities

- `dataset/dataloader.py`: dataset indexing and reading individual images with original annotations. Add PyTorch batching only when needed.
- `dataset/prepro.py`: animal filtering, selection CSV, and later crop preparation with coordinate metadata.
- `model/`: pretrained model loading and inference wrappers, created when integrating the pose model.
- `run/inference.py`: inference orchestration, created after dataset selection.
- `run/evaluation.py`: evaluation orchestration, created when reference annotations are available.
- `run/trainer.py`: optional future fine-tuning; not implemented during feasibility testing.
- `runs/`: generated results, excluded from Git along with `data/` and local model weights.

COCO bounding boxes use `[x, y, width, height]` in original image pixels. Preserve instance annotation IDs and original image dimensions throughout preprocessing.
