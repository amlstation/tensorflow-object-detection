import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from definitions import MODELS_PATH


class Model:
    def __init__(self, model_name):
        self.interpreter = tflite.Interpreter(
            os.path.join(MODELS_PATH, model_name, "saved_model", "model.tflite")
        )
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()[0]["shape"]
        self.input_height = input_details[1]
        self.input_width = input_details[2]

    def normalize_image(self, img_raw):
        image_resized = cv2.resize(img_raw, (self.input_height, self.input_width)) / 255
        img_np = np.expand_dims(image_resized, axis=0).astype(np.float32)

        return img_np

    def set_input_tensor(self, image):
        tensor_index = self.interpreter.get_input_details()[0]["index"]
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details["index"]))
        return tensor

    def detect_objects(self, image):
        self.set_input_tensor(image)
        self.interpreter.invoke()

        scores = self.get_output_tensor(0)
        boxes = self.get_output_tensor(1)
        count = int(self.get_output_tensor(2))
        classes = self.get_output_tensor(3)

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

    def detect_objects_on_img(self, img_raw):
        normalized_img = self.normalize_image(img_raw)
        detections = self.detect_objects(normalized_img)

        return detections
