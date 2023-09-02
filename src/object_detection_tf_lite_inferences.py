import os
import glob
import json
import argparse
import cv2

import numpy as np
import tflite_runtime.interpreter as tflite

from definitions import IMG_EXTENSIONS
from object_detection.utils import label_map_util

global MODELS_PATH
MODELS_PATH = "/workspace/models"


class Model:
    def __init__(self, model_name):
        self.interpreter = tflite.Interpreter(
            os.path.join(MODELS_PATH, model_name, "saved_model", "model.tflite")
        )
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()[0]["shape"]
        self.input_height = input_details[1]
        self.input_width = input_details[2]


def normalize_image(img_raw, detection_model_input_height, detection_model_input_width):
    image_resized = (
        cv2.resize(img_raw, (detection_model_input_height, detection_model_input_width))
        / 255
    )
    img_np = np.expand_dims(image_resized, axis=0).astype(np.float32)

    return img_np


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor


def detect_objects(interpreter, image):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    scores = get_output_tensor(interpreter, 0)
    boxes = get_output_tensor(interpreter, 1)
    count = int(get_output_tensor(interpreter, 2))
    classes = get_output_tensor(interpreter, 3)

    detections = []
    for i in range(count):
        try:
            result = {
                "bounding_box": boxes[i].tolist(),
                "class_id": classes[i],
                "score": scores[i],
            }
            detections.append(result)
        except:
            pass
    return detections


def detect_objects_on_img_file(detection_model, img_raw):
    normalized_img = normalize_image(
        img_raw, detection_model.input_height, detection_model.input_width
    )

    detections = detect_objects(detection_model.interpreter, normalized_img)

    return detections


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
    detections = detect_objects_on_img_file(detection_model, img)

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
