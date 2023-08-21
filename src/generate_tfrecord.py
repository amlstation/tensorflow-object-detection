""" Sample TensorFlow ANNOTATION-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-i IMGS_PATHS_FILE] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i IMGS_PATHS_FILE, --imgs_paths_file IMGS_PATHS_FILE
                        Path to the TXT with the paths of the input images.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -f ANNOTATION_FORMAT, --annotation_format
                        Dataset annotation format: pascal_voc or yolo.
  -c CLASSES, --classes
                        Classes for which the model will be trained.
"""

import os
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
import cv2

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow ANNOTATION-to-TFRecord converter"
)
parser.add_argument(
    "-i",
    "--imgs_paths_file",
    help="Path to the TXT with the paths of the input images.",
    type=str,
)
parser.add_argument(
    "-l", "--labels_path", help="Path to the labels (.pbtxt) file.", type=str
)
parser.add_argument(
    "-o", "--output_path", help="Path of output TFRecord (.record) file.", type=str
)
parser.add_argument(
    "-f",
    "--annotation_format",
    help="Dataset annotation format: pascal_voc or yolo",
    type=str,
    default=None,
)
parser.add_argument(
    "-c",
    "--classes",
    type=str,
    nargs="+",
    help="Classes for which the model will be trained.",
)
args = parser.parse_args()


label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def get_imgs_paths(imgs_paths_file):
    with open(f"{imgs_paths_file}") as f:
        dataset_path_list = imgs_paths_file.split("/")[:-1]
        dataset_path = f"{'/'.join(dataset_path_list)}"
        imgs_paths = [os.path.join(dataset_path, x.strip()) for x in f.readlines()]

    return imgs_paths


def txt_to_csv(imgs_paths_file, classes):
    imgs_paths = get_imgs_paths(imgs_paths_file)
    txt_list = []
    unlabeled_imgs = []
    for img_path in imgs_paths:
        annotation_path = f"{os.path.splitext(img_path)[0]}.txt"

        try:
            with open(annotation_path, "r") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            unlabeled_imgs.append(img_path)
            continue

        img = cv2.imread(f"{img_path}")
        img_height, img_width, _ = img.shape

        for line in lines:
            fields = line.split(" ")

            class_id = int(fields[0])
            center_x = float(fields[1])
            center_y = float(fields[2])
            width = float(fields[3])
            height = float(fields[4])

            xmin = int((center_x - width / 2) * img_width)
            ymin = int((center_y - height / 2) * img_height)
            xmax = int((center_x + width / 2) * img_width)
            ymax = int((center_y + height / 2) * img_height)

            value = (
                img_path,
                int(xmax - xmin),
                int(ymax - ymin),
                classes[class_id],
                xmin,
                ymin,
                xmax,
                ymax,
            )

            txt_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    df = pd.DataFrame(txt_list, columns=column_name)
    print(
        f"[!] Found {len(unlabeled_imgs)} unlabeled images on the dataset. "
        f"Details: {unlabeled_imgs}"
    )

    return df


def xml_to_csv(imgs_paths_file):
    imgs_paths = get_imgs_paths(imgs_paths_file)
    xml_list = []
    unlabeled_imgs = []
    for img_path in imgs_paths:
        xml_file = f"{os.path.splitext(img_path)[0]}.xml"
        try:
            tree = ET.parse(xml_file)
        except FileNotFoundError:
            unlabeled_imgs.append(img_path)
            continue
        root = tree.getroot()
        for member in root.findall("object"):
            value = (
                img_path,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print(
        f"[!] Found {len(unlabeled_imgs)} unlabeled images on the dataset. "
        f"Details: {unlabeled_imgs}"
    )
    return xml_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(group):
    with tf.gfile.GFile(group.filename, "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"]))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(args.output_path)

    if args.annotation_format == "pascal_voc":
        examples = xml_to_csv(args.imgs_paths_file)
    elif args.annotation_format == "yolo":
        examples = txt_to_csv(args.imgs_paths_file, args.classes)

    grouped = split(examples, "filename")
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("[x] Successfully created the TFRecord file: {}".format(args.output_path))


if __name__ == "__main__":
    tf.app.run()
