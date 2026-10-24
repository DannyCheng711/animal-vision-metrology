"""Compare geometric and SAM eye masks using predicted dog boxes.

Run from the project root: python -m run.eye_segmentation_evaluation
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset.dataloader import CocoDataset
from dataset.eye_annotations import load_eye_annotations
from measurement import polygon_mask
from model.animal_segmentation import (
    AnimalSegmenter, download_checkpoint as download_animal_checkpoint,
    load_model as load_animal_model,
)
from model.eye_localization import (
    EyeLocalizer, download_configs, download_checkpoint, model_paths, load_model,
)
from model.eye_segmentation import (
    EyeSegmenter, download_checkpoint as download_sam_checkpoint,
    load_model as load_sam_model,
)
from run.eye_localization_evaluation import match_boxes


def score_mask(result, reference):
    """Return the mask, status, and IoU; failed predictions score zero."""
    # Missing detections and invalid masks receive zero IoU.
    if result is None:
        return np.zeros_like(reference), "missing_detection", 0.0
    mask = np.asarray(result["mask"])
    if mask.shape != reference.shape or mask.dtype != np.bool_:
        return np.zeros_like(reference), "invalid_mask", 0.0
    status = result["status"]
    if status != "ok" or not mask.any():
        return mask, status if status != "ok" else "empty_mask", 0.0
    # Compare the full predicted mask with the manual reference mask.
    intersection = np.logical_and(mask, reference).sum()
    union = np.logical_or(mask, reference).sum()
    return mask, "ok", float(intersection / union)  # Mask IoU = intersection / union.


def draw_outline(draw, mask, color):
    """Draw external contours of a binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(map(int, point)) for point in contour.reshape(-1, 2)]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=color, width=1)


def evaluate(segmenter, dataset, dogs_by_image, inferred_images, output_dir, use_sam=False):
    # ====================================================
    # Step 1: Predict masks using the selected method
    # ====================================================
    output_dir = Path(output_dir)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)

    rows = []
    missing_reference_masks = 0
    for image_id, dogs in dogs_by_image.items():
        metadata = dataset.images[image_id]
        path = dataset.image_dir / metadata["file_name"]
        # Both methods reuse the same predicted keypoints and dog matches.
        points, matches = inferred_images[image_id]
        # Generate geometric or SAM masks for the predicted eyes.
        predictions = segmenter.predict(path, points, use_sam=use_sam) if points else []
        if len(predictions) != len(points):
            raise ValueError("Eye mask count differs from input dog count")
        with Image.open(path) as image:
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        draw.text((5, 5), "Green: reference | Red: prediction", fill="white")

        # ====================================================
        # Step 2: Compare masks with reference polygons and save results
        # ====================================================
        for dog in dogs:
            # Find the matched box for this reference dog.
            index = matches.get(dog["annotation_id"])
            for side, eye in dog["eyes"].items():
                polygon = eye["polygon"]

                # Require at least three vertices for the reference polygon.
                if polygon is None or len(polygon) < 3:
                    missing_reference_masks += 1
                    continue

                # Convert the reference polygon to an original-size binary mask.
                reference = polygon_mask(polygon, overlay.size)
                if not reference.any():
                    missing_reference_masks += 1
                    continue

                # Reference dogs without a matched prediction receive zero IoU.
                result = predictions[index][side] if index is not None else None
                mask, status, iou = score_mask(result, reference)
                name = f"{path.stem}_{dog['annotation_id']}_{side}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "masks" / name)
                draw_outline(draw, reference, "lime")
                draw_outline(draw, mask, "red")
                rows.append({
                    "image_id": image_id, "file_name": metadata["file_name"],
                    "dog_id": dog["dog_id"], "annotation_id": dog["annotation_id"],
                    "side": side, "prediction_index": index, "status": status,
                    "iou": iou, "sam_quality": result["quality"] if result else None,
                    "mask_file": f"masks/{name}",
                })

        overlay.save(output_dir / "overlays" / (path.stem + ".png"))

    # ====================================================
    # Step 3: Calculate mean IoU and mask coverage
    # ====================================================
    if not rows:
        raise ValueError("No eligible reference eye masks")

    # Mean IoU includes failures as zero; coverage counts valid, nonempty masks.
    valid = sum(row["status"] == "ok" for row in rows)
    summary = {
        "method": "sam" if use_sam else "geometric", "use_sam": use_sam,
        "detector": "yolo26x", "localizer": "hrnet-w48",
        "detection_confidence": 0.5, "bbox_match_iou": 0.5, "radius_ratio": 0.15,
        "images": len(dogs_by_image), "reference_eyes": len(rows), "valid_masks": valid,
        "failed_masks": len(rows) - valid, "coverage": valid / len(rows),
        "mean_iou": mean(row["iou"] for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "excluded_missing_reference_masks": missing_reference_masks,
    }

    # ====================================================
    # Step 4: Export per-eye results and the evaluation summary
    # ====================================================
    with (output_dir / "eye_mask_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    project_root = Path(".")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = CocoDataset(project_root / "data/coco/annotations/instances_val2017.json",
                          project_root / "data/coco/val2017")

    # ====================================================
    # Step 1: Load reference eyes and group dogs by image
    # ====================================================
    dogs_by_image = defaultdict(list)
    for dog in load_eye_annotations(project_root / "data/cvat/dogs_min2/coco_dogs_eyes.zip", dataset):
        dogs_by_image[dog["image_id"]].append(dog)

    # ====================================================
    # Step 2: Load the selected segmentation and localization models
    # ====================================================
    # Load Animal Segmentation Model
    detector = AnimalSegmenter(load_animal_model(download_animal_checkpoint("yolo26x")), device=device)
    # Load Eye Localization Model
    download_configs("hrnet-w48")
    download_checkpoint("hrnet-w48")
    config, checkpoint = model_paths("hrnet-w48")
    localizer = EyeLocalizer(load_model(config, checkpoint, device=device))

    # ====================================================
    # Step 3: Predict keypoints and match dogs for scoring
    # ====================================================
    inferred_images = {}
    for image_id in dogs_by_image:
        metadata = dataset.images[image_id]
        path = dataset.image_dir / metadata["file_name"]
        animals = detector.predict(path, confidence=0.5)
        boxes = np.asarray([animal["bbox"] for animal in animals], dtype=np.float32).reshape(-1, 4)
        points = localizer.predict(path, boxes)
        if len(points) != len(animals):
            raise ValueError("Keypoint count differs from predicted box count")
        # Match references for scoring; HRNet uses predicted boxes.
        references = [a for a in dataset.annotations_by_image[image_id]
                      if dataset.categories[a["category_id"]] == "dog" and not a.get("iscrowd", 0)]
        inferred_images[image_id] = (points, match_boxes(animals, references))

    # ====================================================
    # Step 4: Compare geometric and SAM masks using the same keypoints
    # ====================================================
    root = project_root / "outputs/eye_segmentation/yolo26x_hrnet-w48"
    # Geometric masks need no SAM model.
    segmenter = EyeSegmenter()
    comparison = []
    for use_sam in (False, True):
        # Select one method per run; SAM does not fall back to the ellipse.
        if use_sam:
            segmenter = EyeSegmenter(load_sam_model(download_sam_checkpoint(), device=device))

        method = "sam" if use_sam else "geometric"
        summary = evaluate(segmenter, dataset, dogs_by_image, inferred_images, root / method, use_sam)
        comparison.append({key: summary[key] for key in (
            "method", "mean_iou", "coverage", "reference_eyes", "valid_masks", "failed_masks")})

    # ====================================================
    # Step 5: Export the method comparison
    # ====================================================
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
