"""Segment individual animals with a COCO-pretrained YOLO11 model."""
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from dataset.dataloader import CocoDataset
from dataset.prepro import select_dog_images

PROJECT_ROOT = Path(".")
CHECKPOINTS = {"s": "yolo11s-seg.pt"}
CHECKPOINT_SOURCE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def download_checkpoint(architecture="s"):
    destination = PROJECT_ROOT / "weights" / CHECKPOINTS[architecture]
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    with urlopen(f"{CHECKPOINT_SOURCE}/{destination.name}", timeout=120) as response:
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

    def predict(self, image_path, animal_names=("dog",), confidence=0.25):
        # Step 1: Resolve names using YOLO metadata, not COCO annotation IDs.
        class_ids = []
        for name in animal_names:
            matches = [index for index, label in self.model.names.items() if label == name]
            if not matches:
                raise ValueError(f"Unknown model category: {name}")
            class_ids.extend(matches)
        if not class_ids:
            return []

        # Step 2: Keep masks in original-image coordinates, without letterboxing.
        result = self.model.predict(
            source=str(image_path), classes=class_ids, conf=confidence,
            device=self.device, retina_masks=True, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.masks is None:
            raise ValueError("Segmentation model returned boxes without masks")

        predictions = []
        for box, mask in zip(result.boxes, result.masks.data, strict=True):
            # Step 3: Convert YOLO xyxy boxes to top-left xywh for EyeLocalizer.
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


if __name__ == "__main__":
    # Step 1: Select the same COCO images with at least two non-crowd dogs.
    coco_root = PROJECT_ROOT / "data" / "coco"
    dataset = CocoDataset(
        coco_root / "annotations" / "instances_val2017.json",
        coco_root / "val2017",
    )
    selected_images = select_dog_images(dataset, min_dogs=2)

    # Step 2: Load the pretrained model once.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = download_checkpoint("s")
    segmenter = AnimalSegmenter(load_model(checkpoint), device=device)
    output_dir = PROJECT_ROOT / "outputs" / "animal_segmentation" / "yolo11s"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    # Step 3: Export masks, predicted boxes, and visualizations for each image.
    rows = []
    for selected in selected_images:
        image_path = dataset.image_dir / selected["file_name"]
        predictions = segmenter.predict(image_path)
        with Image.open(image_path) as image:
            overlay = np.asarray(image.convert("RGB")).copy()
        instances = []
        for index, prediction in enumerate(predictions):
            mask = prediction["mask"]
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
