"""Evaluate keypoint distances after YOLO26x segmentation and HRNet-w48.

Run from the project root: python -m run.eye_distance_evaluation
"""
import csv
import json
from collections import defaultdict
from itertools import combinations
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
    EyeLocalizer, download_configs, download_checkpoint as download_pose_checkpoint,
    load_model as load_pose_model, model_paths,
)
from run.eye_localization_evaluation import match_boxes


def evaluate_distances(eyes_by_image, output_dir):
    # ====================================================
    # Step 1: Select reference pairs for both distance tasks
    # ====================================================
    rows = []
    for image_id, dogs in eyes_by_image.items():
        pairs = []
        # Keep all reference pairs, including those with missing predictions.
        both_eyes = [eyes for eyes in dogs.values() if "left" in eyes and "right" in eyes]
        for eyes in both_eyes:
            pairs.append(("intra_animal", eyes["left"], eyes["right"]))
        # Pair right eyes from different dogs in the same image.
        right_eyes = [eyes["right"] for eyes in dogs.values() if "right" in eyes]
        # get two dogs combinations
        for first, second in combinations(right_eyes, 2):
            pairs.append(("inter_animal_right", first, second))

        # ====================================================
        # Step 2: Measure valid pairs and compare with reference distances
        # ====================================================
        # kind: intra_animal / inter_animal_right
        for kind, first, second in pairs:
            target = point_distance(first["target"], second["target"]) # ground truth
            # initialize
            predicted = error = relative_error = None
            status = "missing_keypoint"
            if first["valid"] and second["valid"]:
                # get predicted distance
                predicted = point_distance(first["predicted"], second["predicted"])
                error = abs(predicted - target) # absolute error
                # A zero reference distance has no defined relative error.
                relative_error = error / target * 100 if target > 0 else None
                status = "ok"

            rows.append({
                "type": kind, "image_id": image_id, "file_name": first["file_name"],
                "dog_id_a": first["dog_id"], "dog_id_b": second["dog_id"],
                "annotation_id_a": first["annotation_id"], "annotation_id_b": second["annotation_id"],
                "side_a": first["side"], "side_b": second["side"],
                "target_distance_px": target, "predicted_distance_px": predicted,
                "absolute_error_px": error, "relative_error_percent": relative_error,
                "status": status,
            })

    # ====================================================
    # Step 3: Export per-pair distances and errors
    # ====================================================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["type", "image_id", "file_name", "dog_id_a", "dog_id_b", "annotation_id_a",
              "annotation_id_b", "side_a", "side_b", "target_distance_px", "predicted_distance_px",
              "absolute_error_px", "relative_error_percent", "status"]
    with (output_dir / "distance_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ====================================================
    # Step 4: Calculate error and coverage for each distance task
    # ====================================================
    # Errors use measured pairs; coverage includes all reference pairs.
    summary = {}
    for kind in ("intra_animal", "inter_animal_right"):
        eligible = [row for row in rows if row["type"] == kind]
        successful = [row for row in eligible if row["status"] == "ok"]
        relative = [row["relative_error_percent"] for row in successful if row["relative_error_percent"] is not None]
        summary[kind] = {
            "reference_pairs": len(eligible), "measured_pairs": len(successful),
            "missing_pairs": len(eligible) - len(successful),
            "coverage": len(successful) / len(eligible) if eligible else None,
            "mae_px": mean(row["absolute_error_px"] for row in successful) if successful else None,
            "mean_relative_error_percent": mean(relative) if relative else None,
            "relative_error_pairs": len(relative),
        }
    return summary


if __name__ == "__main__":
    project_root = Path(".")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detection_confidence = 0.5

    # ====================================================
    # Step 1: Load reference eyes and group dogs by image
    # ====================================================
    dataset = CocoDataset(project_root / "data/coco/annotations/instances_val2017.json",
                          project_root / "data/coco/val2017")
    annotated_dogs = load_eye_annotations(project_root / "data/cvat/dogs_min2/coco_dogs_eyes.zip", dataset)
    if not annotated_dogs:
        raise ValueError("No valid reference eyes to evaluate")
    dogs_by_image = defaultdict(list)
    for dog in annotated_dogs:
        dogs_by_image[dog["image_id"]].append(dog)

    # ====================================================
    # Step 2: Load the selected segmentation and localization models
    # ====================================================
    # Load Animal Segmentation Model
    animal_model = load_animal_model(download_animal_checkpoint("yolo26x"))
    animal_segmenter = AnimalSegmenter(animal_model, device=device)

    # Load Eye Localization Model
    architecture = "hrnet-w48"
    download_configs(architecture)
    download_pose_checkpoint(architecture)
    config_path, checkpoint_path = model_paths(architecture)
    localizer = EyeLocalizer(load_pose_model(config_path, checkpoint_path, device=device))

    # ====================================================
    # Step 3: Predict eye keypoints and match dogs for scoring
    # ====================================================
    eyes_by_image = defaultdict(dict)
    for image_id, dogs in dogs_by_image.items():
        metadata = dataset.images[image_id]
        image_path = dataset.image_dir / metadata["file_name"]
        # dog segmentation
        animals = animal_segmenter.predict(image_path, confidence=detection_confidence)
        # bbox of dogs
        bboxes = np.asarray([animal["bbox"] for animal in animals], dtype=np.float32).reshape(-1, 4)
        # eye points of dogs
        eye_points = localizer.predict(image_path, bboxes)
        if len(eye_points) != len(animals):
            raise ValueError("Prediction count differs from input box count")

        # Match references for scoring; HRNet uses predicted boxes.
        references = [a for a in dataset.annotations_by_image[image_id]
                      if dataset.categories[a["category_id"]] == "dog" and not a.get("iscrowd", 0)]
        matches = match_boxes(animals, references, threshold=0.5)

        # ====================================================
        # Step 4: Check predicted coordinates for each reference eye
        # ====================================================
        for dog in dogs:
            eyes = {}
            index = matches.get(dog["annotation_id"])
            for side, eye in dog["eyes"].items():
                point = eye_points[index][side + "_eye"] if index is not None else None
                # Use the same finite, in-image check as inference.
                valid = bool(point is not None and np.isfinite(point).all()
                             and 0 <= point[0] < metadata["width"] and 0 <= point[1] < metadata["height"])
                eyes[side] = {"file_name": metadata["file_name"], "dog_id": dog["dog_id"],
                              "annotation_id": dog["annotation_id"], "side": side,
                              "target": eye["point"], "predicted": point, "valid": valid}
            eyes_by_image[image_id][dog["dog_id"]] = eyes

    # ====================================================
    # Step 5: Measure distances (valid eyes) and export the evaluation summary
    # ====================================================
    output_dir = project_root / "outputs/eye_distance/yolo26x_hrnet-w48"
    summary = evaluate_distances(eyes_by_image, output_dir)
    eyes = [eye for dogs in eyes_by_image.values() for dog in dogs.values() for eye in dog.values()]
    summary["settings"] = {"detector": "yolo26x", "localizer": architecture,
                           "detection_confidence": detection_confidence, "bbox_match_iou": 0.5,
                           "measurement_source": "keypoint", "reference_eyes": len(eyes),
                           "valid_eyes": sum(eye["valid"] for eye in eyes),
                           "eye_coverage": sum(eye["valid"] for eye in eyes) / len(eyes)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Outputs:", output_dir)
