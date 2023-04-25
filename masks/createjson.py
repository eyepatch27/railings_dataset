import cv2
import os
import json
from glob import glob

def create_coco_annotations(input_dir, output_file):
    railing_color = (0, 72, 255)
    image_files = glob(os.path.join(input_dir, "*.png"))

    coco_data = {
        "info": {
            "description": "Railing Dataset",
            "version": "1.0",
            "year": 2023,
        },
        "images": [],
        "annotations": [],
    }

    annotation_id = 1

    for image_id, image_path in enumerate(image_files, 1):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        # Find railing pixels
        railing_mask = cv2.inRange(image, railing_color, railing_color)

        # Find contours
        contours, _ = cv2.findContours(railing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
        })

        for contour in contours:
            # Create polygon from contour
            segmentation = contour.flatten().tolist()

            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Assuming only one category (railings)
                "segmentation": [segmentation],
                "bbox": bbox,
                "iscrowd": 0,
            })

            annotation_id += 1

    # Save COCO-formatted data to JSON file
    with open(output_file, "w") as f:
        json.dump(coco_data, f)

if __name__ == "__main__":
    input_dir = "/path/to/your/images/directory"  # Update this path
    output_file = "coco_annotations.json"

    create_coco_annotations(input_dir, output_file)
