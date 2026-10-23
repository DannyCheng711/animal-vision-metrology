"""Read CVAT eye annotations and match local dog IDs to COCO instances."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def load_eye_annotations(zip_path, dataset):
    """Return matched dogs; only measurable, non-uncertain eyes are evaluated."""
    # Uncertain means the eye position is only a human guess.
    # This evaluation only uses measurable points marked as visible.

    # ====================================================
    # Step 1: Read CVAT annotations from the ZIP file
    # ====================================================
    with ZipFile(zip_path) as archive:
        root = ET.fromstring(archive.read("annotations.xml"))

    # ====================================================
    # Step 2: Index COCO images by file name and get dog category IDs
    # ====================================================
    images = {image["file_name"]: image for image in dataset.images.values()}
    dog_category_ids = {key for key, name in dataset.categories.items() if name == "dog"}
    matched_dogs = []

    # ====================================================
    # Step 3: Match each CVAT image to COCO and check image dimensions
    # ====================================================
    for image in root.findall("image"):
        file_name = Path(image.get("name")).name
        metadata = images[file_name] # Get metadata from the original COCO dataset.

        if (int(image.get("width")), int(image.get("height"))) != (
            metadata["width"], metadata["height"]
        ):
            raise ValueError(f"Image dimensions differ: {file_name}")


        # ====================================================
        # Step 4: Group eye points and masks by CVAT dog_id
        # ====================================================
        # Example: group labels by the image-local dog_id.
        # groups = {
        #     "1": {
        #         "L_Eye": {
        #             "coordinates": [[100, 80]],
        #             "valid": True,
        #             "visibility": "visible",
        #         },
        #         "R_Eye": {
        #             "coordinates": [[120, 82]],
        #             "valid": True,
        #             "visibility": "visible",
        #         },
        #     },
        #     ... 
        # }
        groups = defaultdict(dict)

        # Read all shapes in this image (points and polygons).
        for shape in image:
            label = shape.get("label")
            if label not in ("L_Eye", "R_Eye", "L_Eye_Mask", "R_Eye_Mask"):
                continue

            # Read dog_id, visibility, and measurable attributes.
            attributes = {item.get("name"): item.text for item in shape.findall("attribute")}
            dog_id = attributes["dog_id"] # Get the local CVAT dog ID.

            # Read point or polygon coordinates.
            coordinates = [list(map(float, point.split(","))) for point in shape.get("points").split(";")]
            is_point = label in ("L_Eye", "R_Eye")

            # Mark measurable, visible eye points as valid for evaluation.
            groups[dog_id][label] = {
                "coordinates": coordinates,
                "valid": is_point and attributes.get("measurable") == "yes"
                and attributes.get("visibility") in ("visible"),
                "visibility": attributes.get("visibility"),
            }

        # ====================================================
        # Step 5: Collect valid eyes and their masks for each dog
        # ====================================================
        used_annotation_ids = set() # Track COCO annotation IDs already matched in this image.

        for dog_id, labels in groups.items():
            eyes = {}
            for side, label in (("left", "L_Eye"), ("right", "R_Eye")):
                point = labels.get(label)
                if point is not None and point["valid"]:
                    mask = labels.get(label + "_Mask") # Get the matching eye polygon.
                    eyes[side] = {
                        "point": point["coordinates"][0],
                        "visibility": point["visibility"],
                        "polygon": mask["coordinates"] if mask else None,
                    }

            # Skip dogs with no valid eye points.
            if not eyes:
                continue

            # ====================================================
            # Step 6: Match valid eyes to one unique COCO dog segmentation
            # ====================================================
            candidates = []
            for annotation in dataset.annotations_by_image[metadata["id"]]:
                # Only consider non-crowd dog annotations.
                if annotation["category_id"] not in dog_category_ids or annotation.get("iscrowd", 0):
                    continue

                # Get the original COCO dog segmentation, not the eye mask.
                polygons = annotation["segmentation"]
                if not isinstance(polygons, list):
                    raise ValueError(
                        f"Expected polygon segmentation: COCO annotation {annotation['id']}"
                    )

                # Convert each dog polygon to an OpenCV contour of (x, y) points.
                contours = [
                    np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    for polygon in polygons
                ]

                # Each valid eye must be inside or on the edge of a dog polygon.
                # A dog may have multiple polygons, so check any contour per eye.
                inside_all_eyes = True
                for eye in eyes.values():
                    inside_dog = any(
                        cv2.pointPolygonTest(contour, tuple(eye["point"]), False) >= 0
                        for contour in contours
                    )
                    if not inside_dog:
                        inside_all_eyes = False
                        break

                # Keep this dog as a candidate if all valid eyes belong to it.
                if inside_all_eyes:
                    candidates.append(annotation)

            # Require one matching dog; never choose an ambiguous candidate.
            # Raise an error if no dog matches or multiple dogs match.
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot uniquely match {file_name}, dog {dog_id}: "
                    f"found {len(candidates)} candidates"
                )

            # Get the unique matching COCO dog annotation.
            annotation = candidates[0]

            # Make sure this COCO annotation has not been used by another CVAT dog_id.
            if annotation["id"] in used_annotation_ids:
                raise ValueError(
                    f"Multiple dog IDs match COCO annotation {annotation['id']}"
                )

            # Record this COCO annotation ID as used.
            used_annotation_ids.add(annotation["id"])

            # ====================================================
            # Step 7: Save eye ground truth with the matched COCO dog metadata
            # ====================================================
            matched_dogs.append({
                "image_id": metadata["id"],
                "file_name": file_name,
                # Keep both IDs to trace the CVAT-to-COCO match.
                "dog_id": dog_id,
                "annotation_id": annotation["id"],
                "bbox": annotation["bbox"],
                "match_method": "segmentation",
                "eyes": eyes,
            })


    return matched_dogs
