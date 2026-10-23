import csv
import json
from pathlib import Path

from .dataloader import CocoDataset


def select_dog_images(dataset, min_dogs=2):
    """Select images with at least min_dogs non-crowd dog instances."""
   
    # Resolve the category from COCO metadata, not a model's class index.
    dog_category_ids = {
        category_id
        for category_id, name in dataset.categories.items()
        if name == "dog"
    }

    if not dog_category_ids:
        raise ValueError("The annotations do not contain a dog category")

    selected_images = []
    for image_id in dataset.image_ids:
        dog_annotations = [
            annotation
            for annotation in dataset.annotations_by_image[image_id]
            if annotation["category_id"] in dog_category_ids
            # excluding iscrowd in this task (to verify normal case first)
            and annotation.get("iscrowd", 0) == 0 
        ]

        if len(dog_annotations) < min_dogs:
            continue

        metadata = dataset.images[image_id]
        selected_images.append({
            "image_id": image_id,
            "file_name": metadata["file_name"],
            "width": metadata["width"],
            "height": metadata["height"],
            "dog_count": len(dog_annotations),
            "dog_annotations": dog_annotations,
        })

    return selected_images


def export_selection_csv(selected_images, output_path):
    """Write one row per image; annotation IDs link back to original COCO."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # store metadata in csv 
    fieldnames = [
        "image_id", "file_name", "width", "height", "dog_count",
        "dog_annotation_ids",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for selected in selected_images:
            row = {key: selected[key] for key in fieldnames[:-1]}
            row["dog_annotation_ids"] = json.dumps([
                annotation["id"] for annotation in selected["dog_annotations"]
            ])
            writer.writerow(row)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    coco_root = project_root / "data" / "coco"
    dataset = CocoDataset(
        annotation_path=coco_root / "annotations" / "instances_val2017.json",
        image_dir=coco_root / "val2017",
    )

    selected_images = select_dog_images(dataset, min_dogs=2)
    output_path = project_root / "outputs" / "selection" / "two_or_more_dogs.csv"
    export_selection_csv(selected_images, output_path)

    print("Images with at least two dogs:", len(selected_images))
    print("Selected dog instances:", sum(row["dog_count"] for row in selected_images))
    print("Selection CSV:", output_path)
    for selected in selected_images[:5]:
        print(selected["file_name"], "dogs:", selected["dog_count"])
