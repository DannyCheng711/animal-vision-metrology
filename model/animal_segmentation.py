"""Segment individual animals with a COCO-pretrained YOLO segmentation model."""
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from ultralytics import YOLO


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
