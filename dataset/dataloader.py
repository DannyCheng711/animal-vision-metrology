import json
from pathlib import Path
from PIL import Image

class CocoDataset:
    """Read COCO images and annotations without changing image coordinates.

    Start with Python's dataset protocol; a PyTorch Dataset base class can
    be added when training or batching is needed.
    """

    def __init__(self, annotation_path, image_dir):
        self.annotation_path = Path(annotation_path) 
        self.image_dir = Path(image_dir)

        # Read the JSON file using UTF-8 and json.load().
        with self.annotation_path.open("r", encoding="utf-8") as f:
            self.coco = json.load(f)

        # Build dictionaries indexed by the original COCO IDs.
        # self.images: image_id -> image metadata dictionary (id, width, height, file_name)
        # self.categories: category_id -> category name
        self.images = {
            image["id"]: image 
            for image in self.coco["images"]
        }

        self.categories = {
            category["id"]: category["name"]
            for category in self.coco["categories"]
        }

        # Group annotations by image_id.
        # Initialise an empty list for every image, for appending annotations.
        self.annotations_by_image = {
            image_id: []
            for image_id in self.images
        }

        for annotation in self.coco["annotations"]:
            image_id = annotation["image_id"]
            self.annotations_by_image[image_id].append(annotation)

        # Store sorted image IDs for reproducible positional access.
        # A dataset index is a position, not a COCO image ID.
        self.image_ids = sorted(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """Return one sample; annotations retain original pixel coordinates."""
        # Resolve index -> image_id -> image metadata.
        image_id = self.image_ids[index]
        metadata = self.images[image_id]

        # Build image_path from image_dir and metadata['file_name'].
        image_path = self.image_dir / metadata["file_name"]
        # Read the image and convert it to RGB.
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        
        # Return a dictionary with these keys:
        #   image: RGB PIL image (no resize)
        #   image_id: original COCO image ID
        #   image_path: Path to the original image
        #   width, height: original dimensions from image metadata
        #   annotations: list of original annotations for this image
        # COCO bbox format is [x, y, width, height], not [x1, y1, x2, y2].
        annotations = self.annotations_by_image[image_id]
        return {
            "image": image,
            "image_id": image_id,
            "image_path": image_path,
            "width": metadata["width"],
            "height": metadata["height"],
            "annotations": annotations,
        }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    coco_root = project_root / "data" / "coco"

    dataset = CocoDataset(
        annotation_path=coco_root / "annotations" / "instances_val2017.json",
        image_dir=coco_root / "val2017",
    )
    
    # 5,000 images, 36,781 annotations, and 80 categories.
    print("Images:", len(dataset.coco["images"]))
    print("Annotations:", len(dataset.coco["annotations"]))
    print("Categories:", len(dataset.coco["categories"]))
    
    print("Indexed images:", len(dataset.images))
    # categories[17] == "cat".
    print("Category 17:", dataset.categories[17])
    # 36,781 annotations
    print("Grouped annotations:", sum(map(len, dataset.annotations_by_image.values())))
    print("First image IDs:", dataset.image_ids[:5])

    # Dataset length: 5000
    print("Dataset length:", len(dataset))
    sample = dataset[0]
    print("Image ID:", sample["image_id"])
    print("Image size:", sample["image"].size)
    print("Annotations:", len(sample["annotations"]))
    assert sample["image"].size == (sample["width"], sample["height"])
