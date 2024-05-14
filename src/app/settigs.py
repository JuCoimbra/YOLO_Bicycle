import os

import gdown
from git import Repo


class Settings:
    yoloPath: str = os.getenv("YOLO_PATH")
    weightsPath: str = os.getenv("YOLO_WEIGHTS")
    modelPath: str = os.getenv("YOLO_MODEL")
    classesPath: str = os.getenv("YOLO_CLASSES")
    gitYPath: str = os.getenv("GITY_PATH")
    videoPath: str = os.getenv("VIDEO")
    trackedVideoFolderPath: str = os.getenv("TRACKED_VIDEO_FOLDER")
    max_frames: int = int(os.getenv("MAX_FRAMES", 0))


def configureYOLO():
    if not os.path.exists(Settings.yoloPath):
        os.makedirs(os.path.dirname(Settings.yoloPath), exist_ok=True)

    if not os.path.exists(Settings.weightsPath):
        # Baixar os pesos pré-treinados
        gdown.download(
            "https://pjreddie.com/media/files/yolov3.weights",
            Settings.weightsPath,
            quiet=False,
        )

    if not os.path.exists(Settings.gitYPath):
        Repo.clone_from("https://github.com/pjreddie/darknet.git", Settings.gitYPath)


configureYOLO()
