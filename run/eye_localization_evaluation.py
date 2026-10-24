"""Compare eye localization using ground-truth and YOLO26x dog boxes.

Run from the project root: python -m run.eye_localization_evaluation
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from dataset.dataloader import CocoDataset
from dataset.eye_annotations import load_eye_annotations
from measurement import point_distance
from model.animal_segmentation import (
    AnimalSegmenter, download_checkpoint as download_animal_checkpoint,
    load_model as load_animal_model,
)
from model.eye_localization import (
    EyeLocalizer, download_configs, download_checkpoint, model_paths, load_model,
)


def match_boxes(predictions, references, threshold=0.5):
    """Greedy one-to-one matching by bounding-box IoU."""
    candidates = []

    # ====================================================
    # Step 1: Collect box pairs above the IoU threshold
    # ====================================================
    for i, prediction in enumerate(predictions):
        x, y, w, h = prediction["bbox"]

        for j, reference in enumerate(references):
            rx, ry, rw, rh = reference["bbox"]
            intersection = (
                max(0, min(x + w, rx + rw) - max(x, rx))
                * max(0, min(y + h, ry + rh) - max(y, ry))
            )
            union = w * h + rw * rh - intersection
            iou = intersection / union if union > 0 else 0

            if iou >= threshold:
                candidates.append((iou, i, j))

    # ====================================================
    # Step 2: Match each box once, starting with the highest IoU
    # ====================================================
    candidates.sort(reverse=True)
    used_predictions = set()
    used_references = set()
    matches = {}

    for iou, i, j in candidates:
        if i in used_predictions or j in used_references:
            continue
        matches[references[j]["id"]] = i
        used_predictions.add(i)
        used_references.add(j)

    return matches


def evaluate(localizer, dataset, dogs_by_image, detected_images, architecture, source):
    # ====================================================
    # Step 1: Select GT or predicted boxes and localize eyes
    # ====================================================
    rows = []
    for image_id, dogs in dogs_by_image.items():
        metadata = dataset.images[image_id]

        # GT boxes map directly to reference dog IDs.
        if source == "ground_truth":
            animals = [{"bbox": dog["bbox"]} for dog in dogs]
            matches = {dog["annotation_id"]: index for index, dog in enumerate(dogs)}
        # YOLO boxes use IoU matching for scoring only.
        else:
            animals, matches = detected_images[image_id]
        bboxes = np.asarray([animal["bbox"] for animal in animals], dtype=np.float32).reshape(-1, 4)

        # Predict eye coordinates in original-image pixels.
        predictions = localizer.predict(dataset.image_dir / metadata["file_name"], bboxes)
        if len(predictions) != len(animals):
            raise ValueError("Prediction count differs from input box count")

        # ====================================================
        # Step 2: Compare each reference eye with its predicted point
        # ====================================================
        for dog in dogs:
            # Find the matched box for this reference dog.
            index = matches.get(dog["annotation_id"])
            for side, eye in dog["eyes"].items():
                point = predictions[index][side + "_eye"] if index is not None else None
                confidence = predictions[index][side + "_confidence"] if index is not None else None
                # Missing matches and invalid coordinates are recorded separately.
                status = "missing_detection" if index is None else "invalid_point"
                # Coordinate validity does not establish accuracy or visibility.
                if point is not None and np.isfinite(point).all() and 0 <= point[0] < metadata["width"] and 0 <= point[1] < metadata["height"]:
                    status = "ok"

                # Retain failures for coverage calculation.
                rows.append({
                    "image_id": image_id, "file_name": metadata["file_name"],
                    "dog_id": dog["dog_id"],
                    "annotation_id": dog["annotation_id"],
                    "bbox_source": source, "input_index": index,
                    "input_bbox": animals[index]["bbox"] if index is not None else None,
                    "side": side, "target": eye["point"],
                    "predicted": point,
                    "confidence": confidence,
                    "status": status,
                    "error_px": point_distance(eye["point"], point) if status == "ok" else None,
                })

    # ====================================================
    # Step 3: Calculate mean keypoint error and coverage
    # ====================================================
    # Mean error uses valid points; coverage includes all reference eyes.
    errors = [row["error_px"] for row in rows if row["status"] == "ok"]
    summary = {
        "architecture": architecture,
        "bbox_source": source,
        "detection_confidence": 0.5 if source == "yolo26x" else None,
        "bbox_match_iou": 0.5 if source == "yolo26x" else None,
        "images": len(dogs_by_image),
        "reference_eyes": len(rows), "valid_eyes": len(errors),
        "missing_detection_eyes": sum(row["status"] == "missing_detection" for row in rows),
        "invalid_point_eyes": sum(row["status"] == "invalid_point" for row in rows),
        "coverage": len(errors) / len(rows) if rows else None,
        "mean_keypoint_error_px": mean(errors) if errors else None,
        "keypoint_filter": "finite, in-image coordinates; no confidence threshold",
    }

    # ====================================================
    # Step 4: Export per-eye results and the evaluation summary
    # ====================================================
    output = Path(".") / "outputs/eye_localization" / architecture / source
    output.mkdir(parents=True, exist_ok=True)
    with (output / "eye_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return rows, summary


if __name__ == "__main__":
    project_root = Path(".")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = CocoDataset(project_root / "data/coco/annotations/instances_val2017.json",
                          project_root / "data/coco/val2017")

    # ====================================================
    # Step 1: Load reference eyes and group dogs by image
    # ====================================================
    annotated_dogs = load_eye_annotations(project_root / "annotations/coco_dogs_eyes.zip", dataset)
    if not annotated_dogs:
        raise ValueError("No valid reference eyes to evaluate")
    dogs_by_image = defaultdict(list)
    for dog in annotated_dogs:
        dogs_by_image[dog["image_id"]].append(dog)

    # ====================================================
    # Step 2: Reuse the same YOLO detections for both localization models
    # ====================================================
    detector = AnimalSegmenter(load_animal_model(download_animal_checkpoint("yolo26x")), device=device)
    detected_images = {}
    for image_id in dogs_by_image:
        path = dataset.image_dir / dataset.images[image_id]["file_name"]
        # Detect dogs without using reference annotations.
        predictions = detector.predict(path, confidence=0.5)
        # Match against all non-crowd dogs, including those without eye labels.
        references = [a for a in dataset.annotations_by_image[image_id]
                      if dataset.categories[a["category_id"]] == "dog" and not a.get("iscrowd", 0)]

        # Map reference IDs to predicted boxes for scoring only.
        detected_images[image_id] = (predictions, match_boxes(predictions, references))

    comparison = []
    # ====================================================
    # Step 3: Evaluate both models with GT and YOLO boxes
    # ====================================================
    for architecture in ("hrnet-w48", "rtmpose-m"):
        download_configs(architecture)
        download_checkpoint(architecture)
        config, checkpoint = model_paths(architecture)
        localizer = EyeLocalizer(load_model(config, checkpoint, device=device))

        # GT boxes isolate localization error.
        gt_rows, gt = evaluate(localizer, dataset, dogs_by_image, detected_images, architecture, "ground_truth")

        # YOLO boxes include detection and matching failures.
        yolo_rows, yolo = evaluate(localizer, dataset, dogs_by_image, detected_images, architecture, "yolo26x")
        # Compare errors on eyes valid in both runs.
        common = [(gt_row, yolo_row) for gt_row, yolo_row in zip(gt_rows, yolo_rows, strict=True)
                  if gt_row["status"] == "ok" and yolo_row["status"] == "ok"]
        comparison.append({
            "model": architecture,
            "gt_mean_error_px": gt["mean_keypoint_error_px"], "gt_coverage": gt["coverage"],
            "yolo_mean_error_px": yolo["mean_keypoint_error_px"], "yolo_coverage": yolo["coverage"],
            "coverage_drop_percentage_points": (gt["coverage"] - yolo["coverage"]) * 100,
            "common_eyes": len(common),
            "gt_common_mean_error_px": mean(a["error_px"] for a, b in common) if common else None,
            "yolo_common_mean_error_px": mean(b["error_px"] for a, b in common) if common else None,
        })

    # ====================================================
    # Step 4: Export model comparisons and coverage drop
    # ====================================================
    output_path = project_root / "outputs/eye_localization/comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print("Comparison:", output_path)
