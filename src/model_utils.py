import os
import glob

from definitions import MODELS_PATH


def get_latest_checkpoint(model_name: str):
    """Returns the partial path to the latest ckpt- file in the model
    directory

    Raises:
        FileNotFoundError: model directory has no valid ckpt- files"""
    target_dir = f"{MODELS_PATH}/{model_name}"
    ckpt_files = glob.glob(os.path.join(target_dir, "**/ckpt-*"), recursive=True)
    latest_checkpoint = None
    latest_checkpoint_path = None
    for file_path in ckpt_files:
        filename = os.path.basename(file_path)
        ckpt, _ = os.path.splitext(filename)
        ckpt_split = ckpt.split("-")
        ckpt_number = ckpt_split[1] if len(ckpt_split) > 1 else None
        if ckpt_number:
            if latest_checkpoint is None or int(ckpt_number) > int(latest_checkpoint):
                latest_checkpoint = ckpt_number
                latest_checkpoint_path, _ = os.path.splitext(file_path)
    if latest_checkpoint_path is None:
        raise FileNotFoundError(f"Model {model_name} has no valid checkpoint files.")
    return latest_checkpoint_path


def get_checkpoint_by_partial(model_name: str, ckpt_partial: str):
    target_dir = f"{MODELS_PATH}/{model_name}/**/{ckpt_partial}*"
    ckpt_file_path = glob.glob(target_dir, recursive=True)
    if len(ckpt_file_path) == 0:
        raise FileNotFoundError(
            f"couldn't find checkpoint file {ckpt_partial} for model."
        )
    return ckpt_file_path
