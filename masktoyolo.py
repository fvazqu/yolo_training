import os
import cv2
import numpy as np
from pathlib import Path


def mask_to_bbox(mask, min_size=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Only add bounding boxes that are larger than min_size
        if w > min_size and h > min_size:
            bboxes.append((x, y, x + w, y + h))
    return bboxes


def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def save_yolo_annotations(bboxes, class_id, img_width, img_height, output_path):
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def process_dataset(image_dir, mask_dir, output_dir, class_id=0):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.glob('*.jpg'):  # Adjust the extension as needed
        print(f"Processing {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading image {image_path}")
            continue

        mask_path = mask_dir / f"{image_path.stem}.jpg"  # Adjust mask naming as needed
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading mask {mask_path}")
            continue

        img_height, img_width = image.shape[:2]
        bboxes = mask_to_bbox(mask)
        if not bboxes:
            print(f"No bounding boxes found for {image_path}")
            continue

        annotation_path = output_dir / f"{image_path.stem}.txt"
        save_yolo_annotations(bboxes, class_id, img_width, img_height, annotation_path)
        print(f"Saved annotations to {annotation_path}")

name = 'Kvasir-SEG'
cat = 'validation'

image_dir = f'C:path\\{name}\\{cat}\\images'
mask_dir = f'C:path\\{name}\\{cat}\\masks'
output_dir = f'C:path\\{name}\\{cat}\\annotations'


process_dataset(image_dir, mask_dir, output_dir)
