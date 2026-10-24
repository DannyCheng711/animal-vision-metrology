"""Run animal segmentation, eye localization, and eye segmentation on one image.

Run from the project root: python -m run.inference
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

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


def inference(image_path, animal_segmenter, localizer, eye_segmenter, output_dir, use_sam=False):
    """Use predicted dog boxes throughout; no COCO annotations are needed."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)

    # ====================================================
    # Step 1: Segment dogs and use their predicted boxes for HRNet
    # ====================================================
    animals = animal_segmenter.predict(image_path)
    if animals:
        bboxes = np.asarray([animal["bbox"] for animal in animals], dtype=np.float32)
        eye_points = localizer.predict(image_path, bboxes)
        eye_masks = eye_segmenter.predict(image_path, eye_points, use_sam=use_sam)
    else:
        eye_points = []
        eye_masks = []
        print("No dogs detected; eye inference skipped.")

    # ====================================================
    # Step 2: Save animal outlines, eye masks, and mask bounding boxes
    # ====================================================
    predictions = []
    for index, (animal, points, masks) in enumerate(zip(animals, eye_points, eye_masks, strict=True)):
        animal_mask_name = f"dog_{index}.png"
        Image.fromarray(animal["mask"].astype(np.uint8) * 255).save(mask_dir / animal_mask_name)
        draw_mask_outline(draw, animal["mask"], "yellow")
        x, y, width, height = animal["bbox"]
        draw.text((x, max(0, y - 12)), f"dog {index} {animal['confidence']:.2f}", fill="yellow")
        dog = {
            "instance_id": index, "bbox": animal["bbox"],
            "confidence": animal["confidence"], "category": animal["category"],
            "mask_file": f"masks/{animal_mask_name}", "eyes": {},
        }
        for side, color in (("left", "lime"), ("right", "cyan")):
            eye = masks[side]
            mask = eye["mask"]
            mask_name = f"dog_{index}_{side}_eye.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / mask_name)
            draw_mask_outline(draw, mask, color)

            # This is the predicted mask's box, not the SAM prompt box.
            yy, xx = np.nonzero(mask)
            mask_bbox = None
            if len(xx):
                xmin, ymin, xmax, ymax = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())
                mask_bbox = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
            dog["eyes"][side] = {
                "keypoint": points[side + "_eye"],
                "keypoint_confidence": points[side + "_confidence"],
                "mask_bbox": mask_bbox, "mask_file": f"masks/{mask_name}",
                "sam_quality": eye["quality"], "status": eye["status"],
            }
        predictions.append(dog)

    # ====================================================
    # Step 3: Export the original-size visualization and traceable metadata
    # ====================================================
    draw.text((5, 5), "Dog: yellow | Left eye: green | Right eye: cyan", fill="white")
    overlay.save(output_dir / "overlay.png")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump({"image_path": str(image_path), "use_sam": use_sam,
                   "bbox_format": "xywh", "predictions": predictions}, file, indent=2)
    print("Dogs:", len(predictions))
    print("Outputs:", output_dir)
    return predictions


if __name__ == "__main__":
    # Change these two settings to select an image and enable SAM.
    project_root = Path(".")
    image_path = project_root / "data" / "cvat" / "dogs_min2" / "images" / "000000081766.jpg"
    use_sam = True
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load each model once.
    animal_model = load_animal_model(download_animal_checkpoint("yolo26x"))
    animal_segmenter = AnimalSegmenter(animal_model, device=device)

    architecture = "w32"
    download_configs(architecture)
    download_pose_checkpoint(architecture)
    config_path, checkpoint_path = model_paths(architecture)
    localizer = EyeLocalizer(load_pose_model(config_path, checkpoint_path, device=device))

    if use_sam:
        eye_model = load_eye_model(download_eye_checkpoint("vit_b"), device=device)
        eye_segmenter = EyeSegmenter(eye_model)
    else:
        eye_segmenter = EyeSegmenter()

    method = "geometric_sam" if use_sam else "geometric"
    output_dir = project_root / "outputs" / "inference" / image_path.stem / method
    inference(image_path, animal_segmenter, localizer, eye_segmenter, output_dir, use_sam=use_sam)
