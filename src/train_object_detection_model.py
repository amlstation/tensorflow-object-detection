import argparse
import multiprocessing
import os
import sys
import tensorflow as tf
import time

from datetime import datetime
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2, preprocessor_pb2

from definitions import (
    DATASETS_PATH,
    MODELS_PATH,
    TF_OBJECT_DETECTION_API_PATH,
    WORKING_DIR,
)

from dataset_utils import (
    check_dataset_split,
    get_classes_names,
)
from model_utils import get_latest_checkpoint, get_checkpoint_by_partial


class DataAugmentationOptions:
    def __init__(self, new_data_augmentation=None):
        if new_data_augmentation is None:
            new_data_augmentation = preprocessor_pb2.PreprocessingStep()

        self.available_augmentation_options = {
            # "normalize_image": new_data_augmentation.normalize_image,  # Leads to: TypeError: tf__normalize_image() missing 4 required positional arguments
            "random_horizontal_flip": new_data_augmentation.random_horizontal_flip,
            "random_pixel_value_scale": new_data_augmentation.random_pixel_value_scale,
            "random_image_scale": new_data_augmentation.random_image_scale,
            "random_rgb_to_gray": new_data_augmentation.random_rgb_to_gray,
            "random_adjust_brightness": new_data_augmentation.random_adjust_brightness,
            "random_adjust_contrast": new_data_augmentation.random_adjust_contrast,
            "random_adjust_hue": new_data_augmentation.random_adjust_hue,
            "random_adjust_saturation": new_data_augmentation.random_adjust_saturation,
            "random_distort_color": new_data_augmentation.random_distort_color,
            "random_jitter_boxes": new_data_augmentation.random_jitter_boxes,
            "random_crop_image": new_data_augmentation.random_crop_image,
            "random_pad_image": new_data_augmentation.random_pad_image,
            "random_crop_pad_image": new_data_augmentation.random_crop_pad_image,
            "random_crop_to_aspect_ratio": new_data_augmentation.random_crop_to_aspect_ratio,
            "random_black_patches": new_data_augmentation.random_black_patches,
            # "random_resize_method": new_data_augmentation.random_resize_method,  # Leads to: error about "output dimensions must be positive"
            "scale_boxes_to_pixel_coordinates": new_data_augmentation.scale_boxes_to_pixel_coordinates,
            # "resize_image": new_data_augmentation.resize_image  # Leads to: error about "output dimensions must be positive"
            # "subtract_channel_mean": new_data_augmentation.subtract_channel_mean,  # Leads to: TypeError: object of type 'NoneType' has no len()
            "ssd_random_crop": new_data_augmentation.ssd_random_crop,
            "ssd_random_crop_pad": new_data_augmentation.ssd_random_crop_pad,
            "ssd_random_crop_fixed_aspect_ratio": new_data_augmentation.ssd_random_crop_fixed_aspect_ratio,
            "ssd_random_crop_pad_fixed_aspect_ratio": new_data_augmentation.ssd_random_crop_pad_fixed_aspect_ratio,
            "random_vertical_flip": new_data_augmentation.random_vertical_flip,
            "random_rotation90": new_data_augmentation.random_rotation90,
            # "rgb_to_gray": new_data_augmentation.rgb_to_gray  Leads to: Input 0 of layer "Conv1" is incompatible with the layer
            # "convert_class_logits_to_softmax": new_data_augmentation.convert_class_logits_to_softmax,  # Leads to: ValueError: Tried to convert 'x' to a tensor and failed. Error: None values not supported.
            "random_absolute_pad_image": new_data_augmentation.random_absolute_pad_image,
            "random_self_concat_image": new_data_augmentation.random_self_concat_image,
            # "autoaugment_image": new_data_augmentation.autoaugment_image,  # Leads to: NameError: name 'contrib_training' is not defined
            "drop_label_probabilistically": new_data_augmentation.drop_label_probabilistically,
            "remap_labels": new_data_augmentation.remap_labels,
            "random_jpeg_quality": new_data_augmentation.random_jpeg_quality,
            "random_downscale_to_target_pixels": new_data_augmentation.random_downscale_to_target_pixels,
            "random_patch_gaussian": new_data_augmentation.random_patch_gaussian,
            "random_square_crop_by_scale": new_data_augmentation.random_square_crop_by_scale,
            "random_scale_crop_and_pad_to_square": new_data_augmentation.random_scale_crop_and_pad_to_square,
            "adjust_gamma": new_data_augmentation.adjust_gamma,
        }


def check_wanted_augmentation(wanted_augmentation, data_augmentation_options):
    if wanted_augmentation not in list(data_augmentation_options.keys()):
        raise ValueError(
            f"{wanted_augmentation} is not a valid data augmentation options."
        )


def create_model_directory(model_name):
    os.makedirs(f"{MODELS_PATH}/{model_name}")
    os.makedirs(f"{ANNOTATIONS_PATH}")


def define_labels(classes):
    labels = []
    label_id_offset = 1  # Label map id 0 is reserved for the background label
    for i, _class in enumerate(classes):
        labels.append({"name": _class, "id": i + label_id_offset})

    return labels


def generate_label_map(labels):
    with open(ANNOTATIONS_PATH + "/label_map.pbtxt", "w") as f:
        for label in labels:
            f.write("item { \n")
            f.write("\tname:'{}'\n".format(label["name"]))
            f.write("\tid:{}\n".format(label["id"]))
            f.write("}\n")


def generate_record(subset, annotation_format, classes):
    classes = str(classes).replace(",", "")[1:-1]
    os.system(
        f"python3 {WORKING_DIR}/src/generate_tfrecord.py \
        -i {DATASET_PATH}/{subset}.txt \
        -l {ANNOTATIONS_PATH}/label_map.pbtxt \
        -o {ANNOTATIONS_PATH}/{subset}.record \
        -f {annotation_format} \
        -c {classes}"
    )


def copy_base_model(model_name, base_model_name):
    os.system(
        f"cp {MODELS_PATH}/{base_model_name}/pipeline.config {MODELS_PATH}/{model_name}"
    )


def remove_unwanted_augmentations_from_pipeline_config(
    wanted_augmentations, pipeline_config
):
    unwated_augmentations = []

    for augmentation in pipeline_config.train_config.data_augmentation_options:
        # The .split(" ")[0] operation is used because augmentation.__str__() contains " {\n}\n" at the end of the string
        # TODO evaluate if better sollution is needed in the future
        matching = [
            s for s in wanted_augmentations if augmentation.__str__().split(" ")[0] in s
        ]
        if len(matching) == 0:
            unwated_augmentations.append(augmentation)

    for unwated_augmentation in unwated_augmentations:
        pipeline_config.train_config.data_augmentation_options.remove(
            unwated_augmentation
        )


def add_wanted_augmentations_to_pipeline_config(wanted_augmentations, pipeline_config):
    new_data_augmentation = preprocessor_pb2.PreprocessingStep()
    data_augmentation_options = DataAugmentationOptions(
        new_data_augmentation
    ).available_augmentation_options

    unwated_augmentations = []
    if len(pipeline_config.train_config.data_augmentation_options) > 0:
        for augmentation in pipeline_config.train_config.data_augmentation_options:
            # The .split(" ")[0] operation is used because augmentation.__str__() contains " {\n}\n" at the end of the string
            # TODO evaluate if better sollution is needed in the future
            augmentation_name = augmentation.__str__().split(" ")[0]
            matching = [s for s in wanted_augmentations if augmentation_name in s]
            if len(matching) != 0:
                unwated_augmentations.append(augmentation_name)

    for wanted_augmentation in wanted_augmentations:
        if wanted_augmentation not in unwated_augmentations:
            data_augmentation_options[wanted_augmentation].SetInParent()
            pipeline_config.train_config.data_augmentation_options.append(
                new_data_augmentation
            )


def update_data_augmentation_options(wanted_augmentations, pipeline_config):
    remove_unwanted_augmentations_from_pipeline_config(
        wanted_augmentations, pipeline_config
    )
    add_wanted_augmentations_to_pipeline_config(wanted_augmentations, pipeline_config)


def configure_pipeline(
    labels,
    input_shape,
    base_model_name,
    check_point,
    num_train_steps,
    warmup_learning_rate,
    warmup_steps,
    batch_size,
    learning_rate,
    wanted_augmentations,
):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(PIPELINE_CONFIG_PATH, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = batch_size
    pipeline_config.train_config.num_steps = num_train_steps
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    pipeline_config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = (
        1e-01
    )
    pipeline_config.model.ssd.post_processing.batch_non_max_suppression.iou_threshold = (
        0.25
    )
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = (
        learning_rate
    )
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = (
        num_train_steps
    )
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = (
        warmup_learning_rate
    )
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = (
        warmup_steps
    )

    # fine_tune_checkpoint expects full path to ckpt- file without the extension
    pipeline_config.train_config.fine_tune_checkpoint = (
        f"{MODELS_PATH}/{base_model_name}/{check_point}"
    )
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

    pipeline_config.train_input_reader.label_map_path = (
        f"{ANNOTATIONS_PATH}/label_map.pbtxt"
    )
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        f"{ANNOTATIONS_PATH}/training.record"
    ]
    pipeline_config.eval_input_reader[
        0
    ].label_map_path = f"{ANNOTATIONS_PATH}/label_map.pbtxt"
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        f"{ANNOTATIONS_PATH}/validation.record"
    ]

    update_data_augmentation_options(wanted_augmentations, pipeline_config)

    config_text = text_format.MessageToString(pipeline_config)

    with tf.io.gfile.GFile(PIPELINE_CONFIG_PATH, "wb") as f:
        f.write(config_text)


def run_tf_object_detection_training_script(model_name, num_train_steps):
    print("Training started...")
    os.system(
        f"python3 {TF_OBJECT_DETECTION_API_PATH}/research/object_detection/model_main_tf2.py \
        --model_dir={MODELS_PATH}/{model_name} \
        --pipeline_config_path={PIPELINE_CONFIG_PATH} \
        --num_train_steps={num_train_steps}"
    )


def run_tf_object_detection_evaluation_script(model_name, evaluate_timeout):
    print("Evaluation started...")
    os.system(
        f"CUDA_VISIBLE_DEVICES=-1 \
        python3 {TF_OBJECT_DETECTION_API_PATH}/research/object_detection/model_main_tf2.py \
        --model_dir={MODELS_PATH}/{model_name} \
        --pipeline_config_path={PIPELINE_CONFIG_PATH} \
        --checkpoint_dir={MODELS_PATH}/{model_name} \
        --eval_timeout {evaluate_timeout}"
    )


def export_tflite_graph(model_name):
    os.system(
        f"CUDA_VISIBLE_DEVICES=-1 \
        python3 {TF_OBJECT_DETECTION_API_PATH}/research/object_detection/export_tflite_graph_tf2.py \
        --pipeline_config_path={PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_dir={MODELS_PATH}/{model_name} \
        --output_directory={MODELS_PATH}/{model_name} \
        --max_detections 80"
    )


def convert_graph_to_tflite(model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(
        f"{MODELS_PATH}/{model_name}/saved_model"
    )
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(
        f"{MODELS_PATH}/{model_name}/saved_model/model.tflite", "wb"
    ) as f:
        f.write(tflite_model)


def convert_object_detection_model_to_tflite(model_name):
    export_tflite_graph(model_name)
    convert_graph_to_tflite(model_name)


def train_object_detection_model(
    dataset_name="garage_door_pascal_voc",
    classes=[""],
    output_model_prefix="detection_model",
    base_model_name="mobilenet_ssd",
    check_point="",
    input_shape=[224, 224],
    num_train_steps=50000,
    warmup_learning_rate=0.0,
    warmup_steps=5000,
    batch_size=4,
    learning_rate=0.001,
    wanted_augmentations=[
        "random_horizontal_flip",
        "random_vertical_flip",
        "random_rotation90",
        "random_patch_gaussian",
        "random_crop_image",
        "random_adjust_hue",
        "random_adjust_contrast",
        "random_adjust_saturation",
        "random_adjust_brightness",
        "random_rgb_to_gray",
    ],
    evaluate=1,
    evaluate_timeout=600,
    annotation_format="yolo",
):
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{output_model_prefix}_{time_stamp}"

    global DATASET_PATH
    global PIPELINE_CONFIG_PATH
    global ANNOTATIONS_PATH
    DATASET_PATH = os.path.join(DATASETS_PATH, dataset_name)
    PIPELINE_CONFIG_PATH = os.path.join(MODELS_PATH, model_name, "pipeline.config")
    ANNOTATIONS_PATH = os.path.join(MODELS_PATH, model_name, "annotations")

    # check that dataset has been split into train/eval
    try:
        check_dataset_split(dataset_name)
    except (FileNotFoundError, OSError):
        print(f"[!] Exiting: dataset {dataset_name} missing split.")
        sys.exit(1)

    create_model_directory(model_name)

    # check whether we received the classes as params or
    # can read from classes.txt
    if classes == [""]:
        class_names_file_path = os.path.join(DATASETS_PATH, dataset_name, "classes.txt")
        try:
            classes = get_classes_names(class_names_file_path)
        except FileNotFoundError as error:
            print(f"[!] Exiting: {str(error)}")
            sys.exit(1)
        print(f"[x] Found class names: {', '.join(classes)}")

    # check that received ckpt file exists, or that models has any at all
    if check_point is None or check_point == "":
        try:
            check_point_path = get_latest_checkpoint(base_model_name)
        except FileNotFoundError as error:
            print(f"[!] Exiting: {str(error)}")
            sys.exit(1)
        check_point = os.path.basename(check_point_path)
    else:
        try:
            get_checkpoint_by_partial(base_model_name, check_point)
        except FileNotFoundError as error:
            print(f"[!] Exiting: {str(error)}")
            sys.exit(1)

    if wanted_augmentations == [""]:
        wanted_augmentations = []
    for wanted_augmentation in wanted_augmentations:
        data_augmentation_options = (
            DataAugmentationOptions().available_augmentation_options
        )

        try:
            check_wanted_augmentation(wanted_augmentation, data_augmentation_options)
        except ValueError as error:
            print(f"[!] Exiting: {str(error)}")
            sys.exit(1)

    labels = define_labels(classes)
    generate_label_map(labels)
    generate_record("training", annotation_format, classes)
    generate_record("validation", annotation_format, classes)
    copy_base_model(model_name, base_model_name)

    configure_pipeline(
        labels,
        input_shape,
        base_model_name,
        check_point,
        num_train_steps,
        warmup_learning_rate,
        warmup_steps,
        batch_size,
        learning_rate,
        wanted_augmentations,
    )

    training_process = multiprocessing.Process(
        target=run_tf_object_detection_training_script,
        kwargs={"model_name": model_name, "num_train_steps": num_train_steps},
    )

    evaluation_process = multiprocessing.Process(
        target=run_tf_object_detection_evaluation_script,
        kwargs={"model_name": model_name, "evaluate_timeout": evaluate_timeout},
    )

    training_process.start()

    if evaluate:
        time.sleep(180)
        evaluation_process.start()

    training_process.join()

    convert_object_detection_model_to_tflite(model_name)

    if evaluate:
        evaluation_process.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="garage_door_pascal_voc")
    parser.add_argument("--classes", type=str, nargs="+", default=[""])
    parser.add_argument(
        "--output_model_prefix", type=str, default="object_detection_model"
    )
    parser.add_argument("--base_model_name", type=str, default="mobilenet_ssd")
    parser.add_argument("--check_point", nargs="?", type=str, default="")
    parser.add_argument("--input_shape", type=int, nargs="*", default=[224, 224])
    parser.add_argument("--num_train_steps", type=int, default=50000)
    parser.add_argument("--warmup_learning_rate", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--wanted_augmentations",
        type=str,
        nargs="+",
        default=[
            "random_horizontal_flip",
            "random_vertical_flip",
            "random_rotation90",
            "random_patch_gaussian",
            "random_crop_image",
            "random_adjust_hue",
            "random_adjust_contrast",
            "random_adjust_saturation",
            "random_adjust_brightness",
            "random_rgb_to_gray",
        ],
    )
    parser.add_argument("--evaluate", type=int, default=1)
    parser.add_argument("--evaluate_timeout", type=int, default=600)
    parser.add_argument("--annotation_format", type=str, default="yolo")
    args = parser.parse_args()

    train_object_detection_model(
        dataset_name=args.dataset_name,
        classes=args.classes,
        output_model_prefix=args.output_model_prefix,
        base_model_name=args.base_model_name,
        check_point=args.check_point,
        input_shape=args.input_shape,
        num_train_steps=args.num_train_steps,
        warmup_learning_rate=args.warmup_learning_rate,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wanted_augmentations=args.wanted_augmentations,
        evaluate=args.evaluate,
        evaluate_timeout=args.evaluate_timeout,
        annotation_format=args.annotation_format,
    )


if __name__ == "__main__":
    main()
