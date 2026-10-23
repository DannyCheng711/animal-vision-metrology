# Model Inference

The `model/` directory contains pretrained model wrappers. `EyeLocalizer.load_model()` loads official HRNet-W32/W48 AP-10K checkpoints through MMPose. Eye prediction and both segmentation wrappers remain unimplemented.

## Project Structure

```text
model/
├── __init__.py
├── animal_segmentation.py  # AnimalSegmenter: YOLO instance segmentation
├── eye_localization.py     # EyeLocalizer: AP-10K pose inference
├── eye_segmentation.py     # EyeSegmenter: SAM/SAM2 prompted masks
└── README.md
```

## Parameters

- `checkpoint_path`: local pretrained weights; keep weight files in the ignored root `weights/` directory.
- `config_path`: architecture configuration where required by the selected backend.
- `device`: execution device, validated when the backend is integrated.

Constructors store parameters. `EyeLocalizer.load_model()` builds the architecture, loads weights, validates eye metadata, and returns the model in evaluation mode. `predict()` still raises `NotImplementedError`. See [environment setup](environment_setup.md) for download and loading commands.

Keep COCO filtering and CSV exports in `dataset/`. Keep inference orchestration and original-image coordinate mapping outside model wrappers. Integrate eye localization first; defer other models until feasibility is verified.

## Animal instance segmentation

Run from the project root:

```bash
.venv/bin/python -m model.animal_segmentation
```

`model/animal_segmentation.py` follows the localization wrapper style: `download_checkpoint`, `load_model`, and `AnimalSegmenter(model, device).predict(image_path)`. It uses the COCO-pretrained `yolo11s-seg.pt` checkpoint and defaults to dog predictions at confidence 0.25. Category names are resolved from model metadata; YOLO class IDs are not COCO annotation IDs.

Each prediction returns a top-left `xywh` bbox compatible with `EyeLocalizer`, an original-resolution boolean mask, YOLO class ID, category name, and confidence. No detections returns an empty list. `animal_names` can select other model categories. YOLO receives only the image; reference dog counts are used solely for reporting.

The demo processes all images selected with `min_dogs=2` and writes `predictions.json`, binary `masks/`, and colored `overlays/` to `outputs/animal_segmentation/yolo11s/`. Predicted instance IDs are image-local and have not been matched to COCO annotation IDs. Counts are a visual/debug check, not IoU, Dice, or mAP evaluation.

This module is currently independent of the eye evaluation entry point. Eye evaluation still uses COCO reference boxes; full-pipeline prediction matching and animal-mask evaluation are separate next steps.

Model reference: https://docs.ultralytics.com/models/yolo11/


## Current distance measurement: keypoints

The current evaluation uses HRNet predicted keypoints versus manual CVAT points for both within-dog eye distances and between-dog right-eye distances. SAM remains an independent eye-mask output with IoU/Dice evaluation. Mask failures and missing reference polygons do not remove otherwise valid point measurements. CSV `confidence` contains HRNet confidence; `sam_quality` is separate.

Run `.venv/bin/python -m run.evaluation`. Current outputs are in `outputs/evaluation/w32_sam_vit_b_keypoint/`: `eye_errors.csv`, `distance_errors.csv`, `summary.json`, `masks/`, and `overlays/`. The summary identifies `measurement_source=keypoint`. COCO ground-truth dog boxes are still used for eye localization.


SAM uses each HRNet eye keypoint as a foreground prompt and selects the candidate mask with the highest predicted quality. `EyeSegmenter.predict()` returns each eye's original-resolution binary mask, quality score, and status. The image is encoded once for all eyes. Invalid prompts return an empty mask with `invalid_prompt`; empty predictions have `empty_mask` status. A single-point prompt can select an entire dog instead of an eye, so inspect mask overlays when interpreting IoU/Dice.

`measurement.py` provides Euclidean point distance and reference polygon rasterization. Distance error is `e = predicted_distance - reference_distance`; MAE is `mean(abs(e))` in pixels and MSE is `mean(e^2)` in squared pixels. Each measurement task is summarized separately.


## Eye masks with an optional SAM step

Use one prediction interface:

```python
segmenter = EyeSegmenter()  # No model is needed for geometric masks.
results = segmenter.predict(image_path, eye_predictions, use_sam=False)

segmenter = EyeSegmenter(sam_model)
results = segmenter.predict(image_path, eye_predictions, use_sam=True)
```

Both modes construct the same geometric ellipse from HRNet keypoints. Horizontal radius is `clip(predicted_inter_eye_distance * radius_ratio, 2, 20)` pixels (default ratio 0.15); vertical radius is `max(1, horizontal_radius * 0.75)`. These are heuristic size parameters. The predicted eyes must be finite, in bounds, and distinct.

With `use_sam=False`, return the geometric mask directly, with null quality. With `use_sam=True`, use its geometric bounding box and keypoint to prompt SAM, replace the ellipse with the highest-quality candidate clipped to the box, and keep only the component containing the prompt pixel. The ellipse itself is not passed as SAM mask_input. An empty SAM result is reported explicitly, without silently falling back to the ellipse. No manual mask selects or filters predictions.

In `run/evaluation.py`, set `use_sam=True` or `False`. Disabling SAM skips its model loading and checkpoint download. Summary records the flag and method; results are saved separately by method. Distances always use keypoints, independent of this flag. The former separate prediction methods are replaced by this single interface.

## Single-image inference

Run `.venv/bin/python -m run.inference` from the project root. Change `image_path` and `use_sam` in the entry point. The simple pipeline is YOLO predicted dog masks/boxes → HRNet keypoints → geometric eye masks with optional SAM. No annotation files or reference boxes are used. Disabling SAM skips its loading and download.

Outputs are saved to `outputs/inference/<image_stem>/<geometric_or_geometric_sam>/`: `overlay.png`, `predictions.json`, and binary PNGs in `masks/`. Yellow outlines show dogs; green shows left eyes; cyan shows right eyes. Eye rectangles enclose the final predicted mask, not the SAM prompt box. JSON boxes use top-left xywh; empty eye masks have null boxes and an explicit status. Instance IDs are local predictions, not CVAT IDs. If no dogs are detected, the original image and empty predictions are saved. Different inputs with the same stem share an output directory, so choose a different output_dir when needed.
