import numpy as np
from pathlib import Path
from urllib.request import urlopen
from mmpose.apis import init_model, inference_topdown


PROJECT_ROOT = Path(".")
CONFIG_ROOT = PROJECT_ROOT / "weights" / "mmpose" / "configs"
CONFIG_SOURCE = "https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/configs"


CHECKPOINTS = {
    "hrnet-w48": "hrnet_w48_ap10k_256x256-d95ab412_20211029.pth",
    "rtmpose-m": "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth",
}


def model_paths(architecture):
    if architecture not in CHECKPOINTS:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if architecture == "rtmpose-m":
        config_dir = "animal_2d_keypoint/rtmpose/ap10k"
        config_name = "rtmpose-m_8xb64-210e_ap10k-256x256.py"

    elif architecture == "hrnet-w48":
        config_dir = "animal_2d_keypoint/topdown_heatmap/ap10k"
        config_name = "td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py"

    config_path = CONFIG_ROOT / config_dir / config_name
    checkpoint_path = PROJECT_ROOT / "weights" / CHECKPOINTS[architecture]

    return config_path, checkpoint_path


def download_file(url, destination):
    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=120) as response:
        # Keep partial downloads separate from the final destination.
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        temporary.replace(destination)


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

    if architecture == "rtmpose-m":
        source = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1"
    elif architecture == "hrnet-w48":
        source = "https://download.openmmlab.com/mmpose/animal/hrnet"

    url = f"{source}/{checkpoint_path.name}"
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
    """Return original-image eye coordinates and confidence for each animal box."""

    def __init__(self, model):
        self.model = model

        # Resolve animal eye indices from the model metadata.
        keypoint_names = model.dataset_meta["keypoint_name2id"]

        self.eye_indices = {
            "left_eye": keypoint_names["L_Eye"],
            "right_eye": keypoint_names["R_Eye"],
        }

    def predict(self, image_path, bboxes):
        # ====================================================
        # Step 1: Check for input animal boxes
        # ====================================================
        if len(bboxes) == 0:
            return []

        # ====================================================
        # Step 2: Predict poses using original-image xywh boxes
        # ====================================================
        results = inference_topdown(
            model=self.model,
            img=str(image_path),
            bboxes=bboxes,
            bbox_format="xywh",
        )

        # ====================================================
        # Step 3: Extract eye coordinates and confidence from each pose
        # ====================================================
        predictions = []

        for result in results:
            pose = result.pred_instances

            # Coordinates are in original-image pixels; scores do not verify visibility.
            keypoints = pose.keypoints[0]
            scores = pose.keypoint_scores[0]

            # Left/right refer to the animal's own sides.
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
