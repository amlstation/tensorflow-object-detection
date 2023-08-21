import os
import glob
import pandas as pd

from definitions import DATASETS_PATH, IMG_EXTENSIONS


def split_dataset(
    dataset_name: str,
    train_fraction: float,
    val_fraction: float = None,
    random_state: int = None,
):
    """Randomly splits object detection dataset images into training,
    validation and test (optional).

    Raises:
        FileNotFoundError: `dataset_path` does not exist.
    """
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    print(dataset_path)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found.")

    imgs = []
    for ext in IMG_EXTENSIONS:
        imgs.extend(glob.glob(os.path.join(dataset_path, ext), recursive=True))
    imgs_df = pd.DataFrame(imgs)

    training_imgs_df = imgs_df.sample(frac=train_fraction, random_state=random_state)
    if val_fraction:
        remaining_imgs_df = imgs_df.drop(training_imgs_df.index)

        proportional_val_fraction = val_fraction / (1 - train_fraction)

        validation_imgs_df = remaining_imgs_df.sample(
            frac=proportional_val_fraction, random_state=random_state
        )
        test_imgs_df = remaining_imgs_df.drop(validation_imgs_df.index)
    else:
        validation_imgs_df = imgs_df.drop(training_imgs_df.index)
        test_imgs_df = None

    for df, name in [
        (training_imgs_df, "training"),
        (validation_imgs_df, "validation"),
        (test_imgs_df, "test"),
    ]:
        txt_file_path = os.path.join(dataset_path, f"{name}.txt")
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)

        if df is None:
            continue

        imgs_list = df[0].values.tolist()
        imgs_list = [img_path.replace(f"{dataset_path}/", "") for img_path in imgs_list]
        with open(txt_file_path, "w", encoding="UTF8") as txt:
            txt.write("\n".join(map(str, imgs_list)))


def get_classes_names(classes_names_file_path):
    with open(classes_names_file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def check_dataset_split(dataset_name: str):
    """Raises exception if dataset {dataset_name} doesn't contains
    the necessary validation.txt or training.txt files"""
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        raise OSError(f"Dataset {dataset_name} not found.")

    validation_file = os.path.join(dataset_path, "training.txt")
    training_file = os.path.join(dataset_path, "validation.txt")

    if not os.path.isfile(validation_file) or not os.path.isfile(training_file):
        raise FileNotFoundError(
            f"Dataset {dataset_name} needs to be split before training."
        )
