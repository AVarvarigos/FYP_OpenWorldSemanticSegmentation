import os

import colorsys
import cv2
from src.datasets import Cityscapes

import pathlib


def get_interpolated_bbox(label, target_class):
    """
    Given a label mask and a target class, compute an aggregated bounding box.
    It finds contours for the target class and then takes the min and max of each bounding box.
    Returns (x, y, w, h) or None if target class not found.
    """
    # Create binary mask for target class
    mask = (label == target_class).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Get bounding boxes for each contour
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bboxes


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        # Generate a color in HSV space and convert it to RGB
        hue = i / n  # equally spaced hue values
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # full saturation and value
        rgb = [int(c * 255) for c in rgb]  # Convert to 0-255 range
        colors.append(rgb)
    return colors


def prepare_data(classes):
    dataset_kwargs = {"n_classes": 19}

    # train data
    train_data = Cityscapes(
        data_dir=os.getenv("CITYSCAPES_DATASET_DIR"),
        split="train",
        classes=classes,
        **dataset_kwargs,
    )

    colors = generate_distinct_colors(classes + 1)

    output_dir = pathlib.Path(
        "/workspace/FYP/FYP_OpenWorldSemanticSegmentation/v2/output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_dir = pathlib.Path(
        "/workspace/FYP/FYP_OpenWorldSemanticSegmentation/v2/datasets/class_crops"
    )
    crop_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(train_data)):
        out = train_data[idx]
        img = out["image"]
        label = out["label"]
        name = out["name"]

        crop_width, crop_height = 256, 256

        name = name.replace(
            "/workspace/FYP/FYP_OpenWorldSemanticSegmentation/v2/datasets/cityscapes/",
            "",
        )

        # Image dimensions
        img_h, img_w = img.shape[0], img.shape[1]
        img_area = img_w * img_h

        for i in range(classes + 1):
            bboxes = get_interpolated_bbox(label, i + 1)
            if bboxes is None:
                continue
            for b_i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                if (w * h) > (0.5 * img_area):
                    continue
                # Crop the bounding box from the image
                crop = img[y : y + h, x : x + w]
                # Save the cropped image
                crop_name = (
                    f"class_{i + 1}/{name.split('/')[-1].split('.')[0]}_{b_i}.png"
                )
                crop_path = crop_dir.joinpath(crop_name)
                crop = cv2.resize(
                    crop, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_path), crop)

        for i in range(classes + 1):
            bboxes = get_interpolated_bbox(label, i + 1)
            if bboxes is None:
                continue
            for bbox in bboxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), colors[i], 2)
                cv2.putText(
                    img,
                    Cityscapes.CLASS_NAMES_REDUCED[i],
                    (x + 10, y + 10),
                    0,
                    0.3,
                    (0, 0, 0),
                )
        # save image with bounding boxes
        image_path = output_dir.joinpath(name)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), img)
        if idx > 10:
            break


if __name__ == "__main__":
    prepare_data(19)
