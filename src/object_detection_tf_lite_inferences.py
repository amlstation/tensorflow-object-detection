import os
import glob
import json
import argparse
import cv2

import numpy as np

from definitions import IMG_EXTENSIONS, MODELS_PATH
from object_detection_model import Model
from object_detection.utils import label_map_util


def get_bounding_box_pixels(img, bounding_box):
    ymin, xmin, ymax, xmax = bounding_box
    img_height, img_width, _ = img.shape

    xmin = int(xmin * img_width)
    xmax = int(xmax * img_width)
    ymin = int(ymin * img_height)
    ymax = int(ymax * img_height)

    return [ymin, xmin, ymax, xmax]


def draw_bounding_box_on_image(img, detection):
    ymin, xmin, ymax, xmax = get_bounding_box_pixels(img, detection["bounding_box"])
    bounding_box_color = (255, 0, 0)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bounding_box_color, 3)


def run_detection_model(
    img_path,
    detection_model,
    label_names,
    score_threshold,
    output_directory,
):
    img = cv2.imread(img_path)

    # THIS IS THE OBJECT CONTAINING ALL DETECTIONS
    detections = detection_model.detect_objects_on_img(img)

    # FROM THIS POINT FORWARD THE CODE IS JUST AN EXAMPLE OF HOW TO POST-PROCESS THE DETECTIONS
    output_img_file_name = img_path.split("/")[-1]

    filtered_detections = []
    for detection in detections:
        if detection["score"] >= score_threshold:
            detection["bounding_box"] = detection["bounding_box"]
            classes_index = int(detection["class_id"])
            detection["class_id"] = label_names[classes_index]
            detection["score"] = float(detection["score"])
            filtered_detections.append(detection)

            draw_bounding_box_on_image(img, detection)

    cv2.imwrite(f"{output_directory}/{output_img_file_name}", img)

    json.dump(
        filtered_detections,
        open(os.path.join(output_directory, f"{output_img_file_name[:-4]}.json"), "w"),
        indent=4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mobilenet_ssd")
    parser.add_argument("--input_directory", type=str, default=None, required=True)
    parser.add_argument("--output_directory", type=str, default="detection_output")
    parser.add_argument("--score_threshold", type=float, default=0.4)
    args = parser.parse_args()

    global MODELS_PATH
    MODELS_PATH = "/workspace/models"

    detection_model = Model(args.model_name)

    label_map_path = os.path.join(
        MODELS_PATH, args.model_name, "annotations", "label_map.pbtxt"
    )
    label_map = label_map_util.create_category_index_from_labelmap(
        label_map_path, use_display_name=True
    )

    label_names = [item[1]["name"] for item in label_map.items()]

    imgs_paths = []
    for ext in IMG_EXTENSIONS:
        imgs_paths.extend(
            glob.glob(os.path.join(args.input_directory, ext), recursive=True)
        )

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    for img_path in imgs_paths:
        print(img_path)

        run_detection_model(
            img_path,
            detection_model,
            label_names,
            args.score_threshold,
            args.output_directory,
        )


if __name__ == "__main__":
    main()
