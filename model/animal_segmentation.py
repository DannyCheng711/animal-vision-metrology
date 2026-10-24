"""Segment individual animals with a COCO-pretrained YOLO segmentation model."""
import csv
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.dataloader import CocoDataset
from dataset.prepro import select_dog_images

PROJECT_ROOT = Path(".")
CHECKPOINTS = {
    "yolov8x": "yolov8x-seg.pt",
    "yolo11x": "yolo11x-seg.pt",
    "yolo26x": "yolo26x-seg.pt",
}


CHECKPOINT_SOURCES = {
    "yolov8x": "https://github.com/ultralytics/assets/releases/download/v8.3.0",
    "yolo11x": "https://github.com/ultralytics/assets/releases/download/v8.3.0",
    "yolo26x": "https://github.com/ultralytics/assets/releases/download/v8.4.0",
}


def download_checkpoint(architecture="yolo26x"):
    if architecture not in CHECKPOINTS:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose {list(CHECKPOINTS)}")
    destination = PROJECT_ROOT / "weights" / CHECKPOINTS[architecture]
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")  # Keep incomplete downloads separate.
    source = CHECKPOINT_SOURCES[architecture]
    with urlopen(f"{source}/{destination.name}", timeout=120) as response:
        with temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    temporary.replace(destination)
    return destination


def load_model(checkpoint_path):
    model = YOLO(str(checkpoint_path))
    if model.task != "segment":
        raise ValueError("AnimalSegmenter requires a segmentation checkpoint")
    return model


class AnimalSegmenter:
    """Return per-instance masks, xywh boxes, YOLO class IDs, and confidence."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def predict(self, image_path, animal_names=("dog",), confidence=0.25, image_size=640):
        # ====================================================
        # Step 1: Resolve names using YOLO metadata
        # ====================================================
        class_ids = []
        # Get model class IDs for the requested animals.
        for name in animal_names:
            matches = [index for index, label in self.model.names.items() if label == name]
            if not matches:
                raise ValueError(f"Unknown model category: {name}")
            class_ids.extend(matches)
        if not class_ids:
            return []

        # ====================================================
        # Step 2: Predict masks aligned with the original image dimensions.
        # ====================================================
        result = self.model.predict(
            source=str(image_path), classes=class_ids, conf=confidence,
            device=self.device, imgsz=image_size, retina_masks=True, verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.masks is None:
            raise ValueError("Segmentation model returned boxes without masks")

        predictions = []
        for box, mask in zip(result.boxes, result.masks.data, strict=True):
            # ====================================================
            # Step 3: Convert YOLO xyxy boxes to top-left xywh for EyeLocalizer.
            # ====================================================
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls.item())
            binary_mask = mask.cpu().numpy().astype(bool)
            if binary_mask.shape != tuple(result.orig_shape):
                raise ValueError("Predicted mask does not match original image size")
            predictions.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "mask": binary_mask,
                "class_id": class_id,
                "category": self.model.names[class_id],
                "confidence": float(box.conf.item()),
            })
        return predictions


def evaluate_model(dataset, selected_images, architecture, device):
    # ====================================================
    # Step 1: Load this model once and prepare output folders
    # ====================================================
    checkpoint = download_checkpoint(architecture)
    segmenter = AnimalSegmenter(load_model(checkpoint), device=device)
    # Create separate mask and overlay folders for this model.
    output_dir = PROJECT_ROOT / "outputs" / "animal_segmentation" / architecture / "coco_eval"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    # ====================================================
    # Step 2: Evaluate dog masks and export predictions for each image
    # ====================================================
    rows = []
    coco_predictions = []
    category_ids = {name: category_id for category_id, name in dataset.categories.items()}
    for selected in selected_images:
        image_path = dataset.image_dir / selected["file_name"]
        predictions = segmenter.predict(image_path, confidence=0.001)
        with Image.open(image_path) as image:
            overlay = np.asarray(image.convert("RGB")).copy()
        instances = []
        # Export each predicted instance in this image.
        for index, prediction in enumerate(predictions):
            mask = prediction["mask"]
            # Encode the original-size mask for standard COCO segmentation evaluation.
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            encoded["counts"] = encoded["counts"].decode("ascii")
            coco_predictions.append({
                "image_id": selected["image_id"],
                "category_id": category_ids[prediction["category"]],
                "segmentation": encoded,
                "score": prediction["confidence"],
            })
            mask_name = f"{image_path.stem}_{index}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / mask_name)
            color = np.asarray(((index * 83 + 60) % 256, (index * 47 + 180) % 256, (index * 113 + 90) % 256))
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
            instances.append({
                "instance_id": index, "bbox": prediction["bbox"],
                "class_id": prediction["class_id"], "category": prediction["category"],
                "confidence": prediction["confidence"], "mask_file": f"masks/{mask_name}",
            })

        overlay = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay)
        # Draw predicted animal boxes and confidence scores.
        for instance in instances:
            x, y, width, height = instance["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline="yellow", width=2)
            draw.text((x, y), f"dog {instance['instance_id']} {instance['confidence']:.2f}", fill="yellow")
        overlay.save(overlay_dir / (image_path.stem + ".png"))
        rows.append({"image_id": selected["image_id"], "file_name": selected["file_name"],
                     "reference_dog_count": selected["dog_count"], "predictions": instances})
        print(f"{image_path.name}: {len(predictions)} predicted / {selected['dog_count']} reference dogs")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    print("Outputs:", output_dir)

    # ====================================================
    # Step 3: Let COCOeval match masks and calculate AP / AR
    # ====================================================
    with (output_dir / "coco_predictions.json").open("w", encoding="utf-8") as file:
        json.dump(coco_predictions, file)
    reference = COCO(str(dataset.annotation_path))  # Load original ground truth.

    if coco_predictions:
        detected = reference.loadRes(coco_predictions)
    else:
        # loadRes requires a nonempty list; keep empty predictions evaluable.
        detected = COCO()
        detected.dataset = {"images": reference.dataset["images"],
                            "categories": reference.dataset["categories"], "annotations": []}
        detected.createIndex()

    evaluator = COCOeval(reference, detected, iouType="segm")
    evaluator.params.imgIds = [image["image_id"] for image in selected_images]
    evaluator.params.catIds = [category_ids["dog"]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    summary = {
        "architecture": architecture, "images": len(selected_images),
        "category": "dog", "evaluation": "COCOeval segm",
        "image_size": 640, "confidence": 0.001,
        "max_detections": evaluator.params.maxDets,
        "ap_50_95": float(evaluator.stats[0]),
        "ap_50": float(evaluator.stats[1]),
        "ap_75": float(evaluator.stats[2]),
        "ar_100": float(evaluator.stats[8]),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    # ====================================================
    # Step 1: Select the COCO validation images with at least two non-crowd dogs.
    # ====================================================
    coco_root = PROJECT_ROOT / "data" / "coco"
    dataset = CocoDataset(
        coco_root / "annotations" / "instances_val2017.json",
        coco_root / "val2017",
    )
    selected_images = select_dog_images(dataset, min_dogs=2)

    # ====================================================
    # Step 2: Evaluate all three models with the same images and settings
    # ====================================================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    comparison = []
    for architecture in ("yolov8x", "yolo11x", "yolo26x"):
        summary = evaluate_model(dataset, selected_images, architecture, device)
        comparison.append({
            "model": architecture,
            "ap_50_95": summary["ap_50_95"], "ap_50": summary["ap_50"],
            "ap_75": summary["ap_75"], "ar_100": summary["ar_100"],
        })
    output_path = PROJECT_ROOT / "outputs" / "animal_segmentation" / "comparison_coco.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print("Comparison:", output_path)
