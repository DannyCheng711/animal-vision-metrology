"""Segment eyes with geometric ellipses or SAM point and box prompts."""
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from measurement import point_distance

PROJECT_ROOT = Path(".")
CHECKPOINTS = {"vit_b": "sam_vit_b_01ec64.pth"}
CHECKPOINT_SOURCE = "https://dl.fbaipublicfiles.com/segment_anything"


def download_checkpoint(architecture="vit_b"):
    destination = PROJECT_ROOT / "weights" / CHECKPOINTS[architecture]
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")  # Keep incomplete downloads separate.
    with urlopen(f"{CHECKPOINT_SOURCE}/{destination.name}", timeout=120) as response:
        with temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    temporary.replace(destination)
    return destination


def load_model(checkpoint_path, architecture="vit_b", device="cpu"):
    model = sam_model_registry[architecture](checkpoint=str(checkpoint_path))
    model.to(device=device)
    model.eval()
    return model


class EyeSegmenter:
    """Return original-image masks and quality scores for each dog's eyes."""

    def __init__(self, model=None):
        self.predictor = SamPredictor(model) if model is not None else None

    def predict(self, image_path, eye_predictions, use_sam=False, radius_ratio=0.15):
        # ====================================================
        # Step 1: Read the image and prepare the SAM image embedding
        # ====================================================
        if not np.isfinite(radius_ratio) or radius_ratio <= 0:
            raise ValueError("radius_ratio must be positive and finite")
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if use_sam:
            if self.predictor is None:
                raise ValueError("Load a SAM model before enabling use_sam")
            self.predictor.set_image(rgb)

        predictions = []
        for eyes in eye_predictions:
            # ====================================================
            # Step 2: Estimate eye size from predicted keypoints
            # ====================================================
            # Use heuristic radii for the ellipse and SAM box prompt.
            left = np.asarray(eyes["left_eye"], dtype=np.float32)
            right = np.asarray(eyes["right_eye"], dtype=np.float32)

            # Check that both eye coordinates are finite and within the image.
            valid_pair = (
                left.shape == (2,) and right.shape == (2,)
                and np.isfinite(left).all() and np.isfinite(right).all()
                and 0 <= left[0] < width and 0 <= left[1] < height
                and 0 <= right[0] < width and 0 <= right[1] < height
            )
            # Calculate the inter-eye distance in original-image pixels.
            distance = point_distance(left, right) if valid_pair else 0.0
            # Human rule: horizontal radius = inter-eye distance * ratio, clipped to 2-20 px.
            radius_x = float(np.clip(distance * radius_ratio, 2, 20))
            # Human rule: vertical radius = 75% of the horizontal radius, at least 1 px.
            radius_y = max(1.0, radius_x * 0.75)

            # ====================================================
            # Step 3: Check eye coordinates and build box prompts
            # ====================================================
            dog = {}
            for side in ("left", "right"):
                point = np.asarray(eyes[side + "_eye"], dtype=np.float32)
                mask = np.zeros((height, width), dtype=bool)
                # Skip this eye if its point cannot be used as a valid prompt.
                if point.shape != (2,) or not np.isfinite(point).all() or not (
                    0 <= point[0] < width and 0 <= point[1] < height
                ):
                    dog[side] = {"mask": mask, "quality": None, "status": "invalid_prompt"}
                    continue

                # Skip this eye if the pair cannot provide a positive eye scale.
                if not valid_pair or distance <= 0:
                    dog[side] = {"mask": mask, "quality": None, "status": "invalid_eye_scale"}
                    continue

                # Use the ellipse extent as an original-image xyxy box prompt.
                x, y = map(float, point)
                x0 = max(0, int(np.floor(x - radius_x)))
                y0 = max(0, int(np.floor(y - radius_y)))
                x1 = min(width - 1, int(np.ceil(x + radius_x)))
                y1 = min(height - 1, int(np.ceil(y + radius_y)))
                box = np.asarray([x0, y0, x1, y1], dtype=np.float32)
                quality = None  # Geometric masks have no predicted IoU score.

                # ====================================================
                # Step 4: Generate a SAM mask or geometric ellipse
                # ====================================================
                if use_sam:
                    # Return the highest predicted-IoU mask without postprocessing.
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point.reshape(1, 2),
                        point_labels=np.asarray([1], dtype=np.int32),
                        box=box,
                        multimask_output=True,
                    )
                    best = int(np.argmax(scores))
                    quality = float(scores[best])  # SAM predicted IoU, not measured GT IoU.
                    mask = masks[best]
                else:
                    # Baseline: an axis-aligned ellipse centered on the keypoint.
                    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
                    mask[y0:y1 + 1, x0:x1 + 1] = (
                        ((xx - x) / radius_x) ** 2 + ((yy - y) / radius_y) ** 2 <= 1
                    )

                # Store the selected method's mask, quality, and status.
                dog[side] = {
                    "mask": mask, "quality": quality,
                    "status": "ok" if mask.any() else "empty_mask",
                    "box": box.tolist(),
                }

            predictions.append(dog)

        return predictions
