from decouple import config


DATASETS_PATH = config("DATASETS_PATH", default="/workspace/datasets")
MODELS_PATH = config("MODELS_PATH", default="/workspace/models")
WORKING_DIR = config("WORKING_DIR", default="/workspace")
IMG_EXTENSIONS = config("IMG_EXTENSIONS", ("**/*.jpg", "**/*.jpeg", "**/*.png"))
TF_OBJECT_DETECTION_API_PATH = config(
    "TF_OBJECT_DETECTION_API_PATH", default="/workspace/tf_object_detection_api"
)


class CredentialError(ValueError):
    pass
