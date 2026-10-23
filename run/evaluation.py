"""Evaluate SAM eye masks and keypoint distances using COCO dog boxes.

Run from the project root: python -m run.evaluation
"""
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset.dataloader import CocoDataset
from dataset.eye_annotations import load_eye_annotations
from measurement import point_distance, polygon_mask
from model.eye_segmentation import (
    EyeSegmenter, download_checkpoint as download_sam_checkpoint,
    load_model as load_sam_model,
)
from model.eye_localization import (
    EyeLocalizer, download_checkpoint, download_configs, load_model, model_paths,
)


def evaluate_eye_distances(eye_rows, output_dir):
    """Compare predicted keypoint and manual point distances in each image."""
    # Step 1: Group valid eyes by image and local dog ID.
    images = defaultdict(dict)
    for row in eye_rows:
        dog = images[row["file_name"]].setdefault(row["dog_id"], {})
        dog[row["side"]] = row

    distance_rows = []
    for dogs in images.values():
        pairs = []
        # Step 2: Measure both eyes of the same dog only when both are valid.
        for eyes in dogs.values():
            if "left" in eyes and "right" in eyes:
                pairs.append(("within_dog", eyes["left"], eyes["right"]))

        # Step 3: Measure all unique right-eye pairs in the same image.
        right_eyes = [eyes["right"] for eyes in dogs.values() if "right" in eyes]
        for first, second in combinations(right_eyes, 2):
            pairs.append(("between_dogs_right", first, second))

        # Step 4: Compare predicted distance against manual reference distance.
        for kind, first, second in pairs:
            target = point_distance(
                (first["target_x"], first["target_y"]),
                (second["target_x"], second["target_y"]),
            )
            predicted = point_distance(
                (first["predicted_x"], first["predicted_y"]),
                (second["predicted_x"], second["predicted_y"]),
            )
            error = predicted - target
            distance_rows.append({
                "type": kind, "image_id": first["image_id"],
                "file_name": first["file_name"],
                "dog_id_a": first["dog_id"], "dog_id_b": second["dog_id"],
                "annotation_id_a": first["annotation_id"],
                "annotation_id_b": second["annotation_id"],
                "side_a": first["side"], "side_b": second["side"],
                "confidence_a": first["confidence"], "confidence_b": second["confidence"],
                "target_distance_px": target, "predicted_distance_px": predicted,
                "signed_error_px": error, "absolute_error_px": abs(error),
                "squared_error_px2": error ** 2,
            })

    # Step 5: Save individual measurements, then summarize each task separately.
    fields = [
        "type", "image_id", "file_name", "dog_id_a", "dog_id_b",
        "annotation_id_a", "annotation_id_b", "side_a", "side_b",
        "confidence_a", "confidence_b", "target_distance_px", "predicted_distance_px",
        "signed_error_px", "absolute_error_px", "squared_error_px2",
    ]
    with (Path(output_dir) / "distance_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(distance_rows)

    summary = {}
    for kind in ("within_dog", "between_dogs_right"):
        measurements = [row for row in distance_rows if row["type"] == kind]
        summary[kind] = {
            "count": len(measurements),
            "mae_px": mean(row["absolute_error_px"] for row in measurements) if measurements else None,
            "mse_px2": mean(row["squared_error_px2"] for row in measurements) if measurements else None,
        }
    return summary


def evaluate(localizer, segmenter, dataset, matched_dogs, output_dir, use_sam=False):
    """Measure keypoint distances and evaluate eye masks independently."""
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    dogs_by_image = defaultdict(list)
    for dog in matched_dogs:
        dogs_by_image[dog["file_name"]].append(dog)

    rows = []
    for file_name, dogs in dogs_by_image.items():
        image_path = dataset.image_dir / file_name
        bboxes = np.asarray([dog["bbox"] for dog in dogs], dtype=np.float32)
        predictions = localizer.predict(image_path, bboxes)
        segmented = segmenter.predict(image_path, predictions, use_sam=use_sam)
        with Image.open(image_path) as source:
            overlay = source.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        draw.text((5, 5), "Green: annotation; red: prediction", fill="white")

        for dog, prediction, masks in zip(dogs, predictions, segmented, strict=True):
            for side, eye in dog["eyes"].items():
                result = masks[side]
                mask = result["mask"]
                predicted = prediction[side + "_eye"]
                target = eye["point"]
                target_mask = polygon_mask(eye["polygon"], overlay.size) if eye["polygon"] else None
                mask_status = result["status"]
                if target_mask is None or not target_mask.any():
                    mask_status = "missing_reference_mask"
                error = point_distance(target, predicted)
                iou = dice = None
                if target_mask is not None and target_mask.any():
                    intersection = int(np.logical_and(mask, target_mask).sum())
                    union = int(np.logical_or(mask, target_mask).sum())
                    iou = intersection / union
                    dice = 2 * intersection / (int(mask.sum()) + int(target_mask.sum()))
                mask_name = f"{Path(file_name).stem}_{dog['annotation_id']}_{side}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / mask_name)
                rows.append({
                    "image_id": dog["image_id"], "file_name": file_name,
                    "dog_id": dog["dog_id"], "annotation_id": dog["annotation_id"],
                    "match_method": dog["match_method"], "side": side,
                    "visibility": eye["visibility"],
                    "target_x": target[0] if target else None, "target_y": target[1] if target else None,
                    "predicted_x": predicted[0] if predicted else None, "predicted_y": predicted[1] if predicted else None,
                    "confidence": prediction[side + "_confidence"], "sam_quality": result["quality"],
                    "mask_status": mask_status, "iou": iou, "dice": dice, "error_px": error,
                })
                # Draw mask contours and keypoint error independently.
                for binary, color in ((target_mask, "lime"), (mask, "red")):
                    if binary is not None:
                        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            points = [tuple(map(int, point)) for point in contour.reshape(-1, 2)]
                            if len(points) > 1:
                                draw.line(points + [points[0]], fill=color)
                draw.line([tuple(target), tuple(predicted)], fill="yellow", width=1)
                for point, color in ((target, "lime"), (predicted, "red")):
                    x, y = point
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)
                draw.text(tuple(target), f" {dog['dog_id']}:{side}", fill="lime")
        overlay.save(overlay_dir / (Path(file_name).stem + ".png"))
        print(f"Evaluated {file_name}: {len(dogs)} dogs")

    if not rows:
        raise ValueError("No valid eye annotations to evaluate")
    with (output_dir / "eye_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for side in ("all", "left", "right"):
        errors = [row["error_px"] for row in rows if row["error_px"] is not None and (side == "all" or row["side"] == side)]
        summary[side] = {
            "count": len(errors),
            "mean_error_px": mean(errors) if errors else None,
            "median_error_px": median(errors) if errors else None,
        }
    summary["mask_method"] = "geometric_sam" if use_sam else "geometric"
    summary["use_sam"] = use_sam
    summary["measurement_source"] = "keypoint"
    summary["reference_eye_count"] = len(rows)
    summary["failed_or_missing_masks"] = sum(row["mask_status"] != "ok" for row in rows)
    summary["eye_masks"] = {
        "count": sum(row["iou"] is not None for row in rows),
        "mean_iou": mean(row["iou"] for row in rows if row["iou"] is not None) if any(row["iou"] is not None for row in rows) else None,
        "mean_dice": mean(row["dice"] for row in rows if row["dice"] is not None) if any(row["dice"] is not None for row in rows) else None,
    }
    summary["distances"] = evaluate_eye_distances(rows, output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    return rows, summary


if __name__ == "__main__":
    project_root = Path(".")
    coco_root = project_root / "data" / "coco"
    dataset = CocoDataset(
        coco_root / "annotations" / "instances_val2017.json",
        coco_root / "val2017",
    )
    matched_dogs = load_eye_annotations(
        project_root / "data" / "cvat" / "dogs_min2" / "coco_dogs_eyes.zip",
        dataset,
    )
    print("Matched dogs:", len(matched_dogs))
    print("Valid eyes:", sum(len(dog["eyes"]) for dog in matched_dogs))

    architecture = "w32"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    download_configs(architecture)
    download_checkpoint(architecture)
    config_path, checkpoint_path = model_paths(architecture)
    model = load_model(config_path, checkpoint_path, device=device)
    localizer = EyeLocalizer(model)

    use_sam = True  # False: geometric masks only; True: add SAM.
    if use_sam:
        sam_checkpoint = download_sam_checkpoint("vit_b")
        sam_model = load_sam_model(sam_checkpoint, device=device)
        segmenter = EyeSegmenter(sam_model)
    else:
        segmenter = EyeSegmenter()

    mask_method = "geometric_sam" if use_sam else "geometric"
    evaluate(localizer, segmenter, dataset, matched_dogs,
             project_root / "outputs" / "evaluation" / f"{architecture}_sam_vit_b_{mask_method}_keypoint",
             use_sam=use_sam)
