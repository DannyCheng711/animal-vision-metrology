"""Evaluate animal segmentation with COCOeval.

Run from the project root: python -m run.animal_segmentation_evaluation
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.dataloader import CocoDataset
from dataset.prepro import select_dog_images
from model.animal_segmentation import AnimalSegmenter, download_checkpoint, load_model

PROJECT_ROOT = Path(".")


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
