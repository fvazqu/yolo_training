import cv2
import os
from pathlib import Path

def denormalize_bbox(x_center, y_center, width, height, img_width, img_height):
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return x_min, y_min, x_max, y_max

def visualize_annotations(image_dir, annotation_dir, output_dir):
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.glob('*.jpg'):  # Adjust the extension as needed
        image = cv2.imread(str(image_path))
        img_height, img_width = image.shape[:2]

        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            print(f"No annotation file for {image_path}")
            continue

        with open(annotation_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_min, y_min, x_max, y_max = denormalize_bbox(x_center, y_center, width, height, img_width, img_height)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, str(int(class_id)), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_image_path = output_dir / image_path.name
        cv2.imwrite(str(output_image_path), image)
        print(f"Saved visualized image to {output_image_path}")

name = 'Kvasir-SEG'
cat = 'validation'

image_dir = f'C:path\\{name}\\{cat}\\images'
annotation_dir = f'C:path\\{name}\\{cat}\\annotations'
output_dir = f'C:path\\{name}\\{cat}\\visualized'

visualize_annotations(image_dir, annotation_dir, output_dir)

