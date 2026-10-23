import numpy as np
import torch
from pathlib import Path
from urllib.request import urlopen
from mmpose.apis import init_model, inference_topdown

from dataset.dataloader import CocoDataset
from dataset.prepro import select_dog_images

PROJECT_ROOT = Path(".")
CONFIG_ROOT = PROJECT_ROOT / "weights" / "mmpose" / "configs"
CONFIG_DIR = "animal_2d_keypoint/topdown_heatmap/ap10k"

CONFIG_SOURCE = (
    "https://raw.githubusercontent.com/open-mmlab/mmpose/"
    "v1.3.2/configs"
)

CHECKPOINTS = {
    "w32": "hrnet_w32_ap10k_256x256-18aac840_20211029.pth",
    "w48": "hrnet_w48_ap10k_256x256-d95ab412_20211029.pth",
}

CHECKPOINT_SOURCE = "https://download.openmmlab.com/mmpose/animal/hrnet"


def model_paths(architecture):
    if architecture not in CHECKPOINTS:
        raise ValueError(f"Unsupported architecture: {architecture}")

    config_name = f"td-hm_hrnet-{architecture}_8xb64-210e_ap10k-256x256.py"

    config_path = CONFIG_ROOT / CONFIG_DIR / config_name
    checkpoint_path = PROJECT_ROOT / "weights" / CHECKPOINTS[architecture]

    return config_path, checkpoint_path


def download_file(url, destination):
    # If destination file already exists, skip downloading
    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=120) as response:
        # Binary write mode
        with destination.open("wb") as file:
            # Read 1 MB per iteration
            while chunk := response.read(1024 * 1024):
                file.write(chunk)


def download_configs(architecture):
    config_path, _ = model_paths(architecture)

    config_files = [
        "_base_/default_runtime.py",
        config_path.relative_to(CONFIG_ROOT).as_posix(),
    ]

    for relative_path in config_files:
        url = f"{CONFIG_SOURCE}/{relative_path}"
        destination = CONFIG_ROOT / relative_path
        download_file(url, destination)


def download_checkpoint(architecture):
    _, checkpoint_path = model_paths(architecture)

    url = f"{CHECKPOINT_SOURCE}/{checkpoint_path.name}"
    download_file(url, checkpoint_path)

    return checkpoint_path


def load_model(config_path, checkpoint_path, device="cpu"):
    model = init_model(
        config=str(config_path),
        checkpoint=str(checkpoint_path),
        device=device,
    )

    return model


class EyeLocalizer:
    """Predict animal eye keypoints using a pretrained MMPose model."""

    def __init__(self, model):
        self.model = model

        keypoint_names = model.dataset_meta["keypoint_name2id"]

        self.eye_indices = {
            "left_eye": keypoint_names["L_Eye"],
            "right_eye": keypoint_names["R_Eye"],
        }

    def predict(self, image_path, bboxes):
        """Return left/right eye coordinates and confidence for each bbox."""
        results = inference_topdown(
            model=self.model,
            img=str(image_path),
            bboxes=bboxes,
            bbox_format="xywh",
        )

        predictions = []

        for result in results:
            pose = result.pred_instances

            keypoints = pose.keypoints[0]
            scores = pose.keypoint_scores[0]

            left_idx = self.eye_indices["left_eye"]
            right_idx = self.eye_indices["right_eye"]

            prediction = {
                "left_eye": keypoints[left_idx].tolist(),
                "right_eye": keypoints[right_idx].tolist(),
                "left_confidence": float(scores[left_idx]),
                "right_confidence": float(scores[right_idx]),
            }

            predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    

    # Load COCO dataset
    coco_root = PROJECT_ROOT / "data" / "coco"

    dataset = CocoDataset(
        annotation_path=coco_root / "annotations" / "instances_val2017.json",
        image_dir=coco_root / "val2017",
    )

    selected_images = select_dog_images(dataset, min_dogs=2)

    if not selected_images:
        raise ValueError("COCO val contains no non-crowd dog instances")

    # Initialize model
    architecture = "w32"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    download_configs(architecture)
    download_checkpoint(architecture)

    config_path, checkpoint_path = model_paths(architecture)

    model = load_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print("Model:", type(model).__name__)
    print("Backbone:", type(model.backbone).__name__)
    print("Head:", type(model.head).__name__)
    print("Evaluation mode:", not model.training)

    # Initialize EyeLocalizer
    localizer = EyeLocalizer(model)

    for selected in selected_images:
        image_path = dataset.image_dir / selected["file_name"]

        bboxes = np.asarray(
            [annotation["bbox"] for annotation in selected["dog_annotations"]],
            dtype=np.float32,
        )

        print("Image:", image_path)
        print(
            "Dog annotation IDs:",
            [annotation["id"] for annotation in selected["dog_annotations"]],
        )

        # Predict eye coordinates
        predictions = localizer.predict(
            image_path=image_path,
            bboxes=bboxes,
        )

        # Display predictions for each dog
        for annotation, prediction in zip(
            selected["dog_annotations"],
            predictions,
            strict=True,
        ):
            print(f"\nDog annotation ID: {annotation['id']}")
            print("Left eye:", prediction["left_eye"])
            print("Right eye:", prediction["right_eye"])
            print("Left confidence:", prediction["left_confidence"])
            print("Right confidence:", prediction["right_confidence"])


# TODO: Add atomic download using .part
# TODO: Add --download CLI argument