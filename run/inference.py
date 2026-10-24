"""Segment dogs and eyes, measure keypoint distances, and export single-image results.

Run from the project root: python -m run.inference
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from measurement import point_distance

from model.animal_segmentation import (
    AnimalSegmenter, download_checkpoint as download_animal_checkpoint,
    load_model as load_animal_model,
)
from model.eye_localization import (
    EyeLocalizer, download_configs, download_checkpoint as download_pose_checkpoint,
    load_model as load_pose_model, model_paths,
)
from model.eye_segmentation import (
    EyeSegmenter, download_checkpoint as download_eye_checkpoint,
    load_model as load_eye_model,
)


def draw_mask_outline(draw, mask, color):
    """Draw all external contours of an original-size binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(map(int, point)) for point in contour.reshape(-1, 2)]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=color, width=2)


def inference(image_path, animal_segmenter, localizer, eye_segmenter, output_dir, detection_confidence=0.5):
    """Predict dog masks, SAM eye masks, and pixel distances without annotations."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)

    # ====================================================
    # Step 1: Segment dogs, locate eye keypoints, and predict SAM eye masks
    # ====================================================
    animals = animal_segmenter.predict(image_path, confidence=detection_confidence)  # YOLO26x-seg: dog masks and bounding boxes.
    if animals:
        bboxes = np.asarray([animal["bbox"] for animal in animals], dtype=np.float32)
        eye_points = localizer.predict(image_path, bboxes)  # HRNet-w48: eye coordinates in the original image.
        eye_masks = eye_segmenter.predict(image_path, eye_points, use_sam=True)  # SAM ViT-B: eye masks from point and box prompts.
    else:
        eye_points = []
        eye_masks = []
        print("No dogs detected; eye inference skipped.")

    # ====================================================
    # Step 2: Save binary masks, draw outlines, and collect prediction metadata
    # ====================================================
    predictions = []
    for index, (animal, points, masks) in enumerate(zip(animals, eye_points, eye_masks, strict=True)):
        animal_mask_name = f"dog_{index}.png"
        Image.fromarray(animal["mask"].astype(np.uint8) * 255).save(mask_dir / animal_mask_name)
        draw_mask_outline(draw, animal["mask"], "yellow")
        # Position the dog ID and confidence label above its bounding box.
        x, y, width, height = animal["bbox"]
        draw.text((x, max(0, y - 12)), f"dog {index} {animal['confidence']:.2f}", fill="yellow")
        dog = {
            "instance_id": index, "bbox": animal["bbox"],
            "confidence": animal["confidence"], "category": animal["category"],
            "mask_file": f"masks/{animal_mask_name}", "eyes": {},
        }

        # Save each eye mask and draw its outline on the overlay.
        for side, color in (("left", "lime"), ("right", "cyan")):
            eye = masks[side]
            mask = eye["mask"]
            mask_name = f"dog_{index}_{side}_eye.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / mask_name)
            draw_mask_outline(draw, mask, color)

            dog["eyes"][side] = {
                "keypoint": points[side + "_eye"],
                "keypoint_confidence": points[side + "_confidence"],
                "mask_file": f"masks/{mask_name}",
                "sam_quality": eye["quality"], "status": eye["status"],
            }

        predictions.append(dog)

    # ====================================================
    # Step 3: Select dogs with valid eye keypoints for distance measurement
    # ====================================================
    both_eyes = []   # Dog indices with two valid eye keypoints.
    right_eyes = []  # Dog indices with a valid right-eye keypoint.
    for index, dog in enumerate(predictions):
        valid_sides = set()
        for side in ("left", "right"):
            x, y = dog["eyes"][side]["keypoint"]
            # Check coordinate validity; this does not establish eye visibility.
            if np.isfinite([x, y]).all() and 0 <= x < overlay.width and 0 <= y < overlay.height:
                valid_sides.add(side)
        if "right" in valid_sides:
            right_eyes.append(index)
        if {"left", "right"} <= valid_sides:
            both_eyes.append(index)

    # ====================================================
    # Step 4: Measure pixel distances and export two distance views and JSON
    # ====================================================
    with Image.open(image_path) as image:
        intra_image = image.convert("RGB")
        inter_image = image.convert("RGB")

    intra_draw = ImageDraw.Draw(intra_image)
    inter_draw = ImageDraw.Draw(inter_image)
    distances = {"intra_animal": [], "inter_animal_right": []}

    # Draw dog outlines and IDs on each corresponding distance view.
    for indices, distance_draw in ((both_eyes, intra_draw), (right_eyes, inter_draw)):
        for index in indices:
            draw_mask_outline(distance_draw, animals[index]["mask"], "yellow")
            x, y, _, _ = animals[index]["bbox"]
            distance_draw.text((x, max(0, y - 12)), f"dog {index}", fill="yellow", stroke_width=1, stroke_fill="black")

    # Measure the left-to-right eye distance for each eligible dog.
    for index in both_eyes:
        left = predictions[index]["eyes"]["left"]["keypoint"]
        right = predictions[index]["eyes"]["right"]["keypoint"]
        distance = point_distance(left, right)
        # Draw the keypoint connection and label its distance in pixels.
        intra_draw.line([tuple(left), tuple(right)], fill="lime", width=1)
        intra_draw.text(tuple(left), f" {distance:.2f}px", fill="lime", stroke_width=1, stroke_fill="black")
        distances["intra_animal"].append({"dog_id": index, "distance_px": distance})
        print(f"Dog {index}: left-right distance = {distance:.2f} px")

    # Measure right-eye distances for all unique pairs of eligible dogs.
    for first, second in combinations(right_eyes, 2):
        a = predictions[first]["eyes"]["right"]["keypoint"]
        b = predictions[second]["eyes"]["right"]["keypoint"]
        distance = point_distance(a, b)
        # Draw the keypoint connection and label its distance in pixels.
        inter_draw.line([tuple(a), tuple(b)], fill="cyan", width=1)
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        inter_draw.text(midpoint, f"{first}-{second}: {distance:.2f}px", fill="cyan", stroke_width=1, stroke_fill="black")
        distances["inter_animal_right"].append({"dog_id_a": first, "dog_id_b": second, "distance_px": distance})
        print(f"Dogs {first}-{second}: right-right distance = {distance:.2f} px")

    print("Both-eye dogs:", both_eyes)
    print("Right-eye dogs:", right_eyes)

    intra_draw.text((5, 5), "Both-eye predictions: distances (px)", fill="white", stroke_width=1, stroke_fill="black")
    inter_draw.text((5, 5), "Right-eye predictions: distances (px)", fill="white", stroke_width=1, stroke_fill="black")
    intra_image.save(output_dir / "both_eyes_distance.png")
    inter_image.save(output_dir / "right_eyes_distance.png")
    (output_dir / "distances.json").write_text(json.dumps({
        "both_eye_dogs": both_eyes,
        "right_eye_dogs": right_eyes, **distances,
    }, indent=2))

    # ====================================================
    # Step 5: Save the segmentation overlay and prediction metadata
    # ====================================================
    draw.text((5, 5), "Dog: yellow | Left eye: green | Right eye: cyan", fill="white")
    overlay.save(output_dir / "overlay.png")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump({"image_path": str(image_path), "use_sam": True,
                   "bbox_format": "xywh", "detection_confidence": detection_confidence,
                   "predictions": predictions}, file, indent=2)
    print("Dogs:", len(predictions))
    print("Outputs:", output_dir)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dog and eye inference on a COCO image.")
    parser.add_argument(
        "--image-id", default="000000081766",
        help="Exact COCO filename stem (default: 000000081766).",
    )
    args = parser.parse_args()

    project_root = Path(".")
    image_path = project_root / "data" / "coco" / "val2017" / f"{args.image_id}.jpg"
    detection_confidence = 0.5
    if not image_path.is_file():
        parser.error(f"COCO image not found: {image_path}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load Animal Segmentation Model
    animal_model = load_animal_model(download_animal_checkpoint("yolo26x"))
    animal_segmenter = AnimalSegmenter(animal_model, device=device)

    # Load Eye Localization Model
    architecture = "hrnet-w48"
    download_configs(architecture)
    download_pose_checkpoint(architecture)
    config_path, checkpoint_path = model_paths(architecture)
    localizer = EyeLocalizer(load_pose_model(config_path, checkpoint_path, device=device))

    # Load Eye Segmentation Model
    eye_model = load_eye_model(download_eye_checkpoint("vit_b"), device=device)
    eye_segmenter = EyeSegmenter(eye_model)

    output_dir = project_root / "outputs" / "inference" / image_path.stem
    inference(image_path, animal_segmenter, localizer, eye_segmenter, output_dir,
              detection_confidence=detection_confidence)
