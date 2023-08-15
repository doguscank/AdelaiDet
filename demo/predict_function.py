# Copyright (c) Dogus Can Korkmaz, 2023. All Rights Reserved
import argparse
from collections import namedtuple
import glob
import multiprocessing as mp
import os
import time
import warnings

import cv2
import numpy as np
import torch
import tqdm
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger

# Turn off all warning messages
warnings.simplefilter("ignore")

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

PREDICTION = namedtuple("prediction", ["cls", "score", "mask", "bbox"])


class Predictor:
    def __init__(
        self,
        config_file: str = "configs/SOLOv2/R101_3x.yaml",
        weights_file: str = "weights/SOLOv2_R101_3x.pth",
    ) -> None:
        assert os.path.exists(
            config_file
        ), f"Config file '{config_file}' doesn't exist!"

        self.config_file = config_file
        self.weights_file = weights_file
        self._init_cfg()

        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(self.cfg)

    def _init_cfg(self) -> None:
        """Initialize config"""
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.config_file)

        assert os.path.exists(self.weights_file)
        opts = ["MODEL.WEIGHTS", self.weights_file]
        self.cfg.merge_from_list(opts)

        self.cfg.freeze()

    def get_prediction_from_image(self, image: np.ndarray):
        """Use given 'image' as input to predict instances.

        Args:
            image (np.ndarray): Image matrix.
        """
        assert (
            type(image) == np.ndarray
        ), "Given image is not an instance of 'numpy.ndarray' !"

        pred = self.predictor(image)
        
        predictions = []

        for _class, _score, _mask, _bbox in list(
            zip(
                pred["instances"].get_fields()["pred_classes"].cpu().numpy(),
                pred["instances"].get_fields()["scores"].cpu().numpy(),
                pred["instances"].get_fields()["pred_masks"].cpu().numpy(),
                pred["instances"].get_fields()["pred_boxes"].tensor.cpu().numpy(),
            )
        ):
            prediction = PREDICTION(CLASSES[_class], _score, _mask.astype(np.int8), _bbox)
            predictions.append(prediction)
            
        return predictions

    def get_prediction_from_path(self, image_path: str):
        """Use given 'image_path' to read image file to predict instances.

        Args:
            image_path (str): Path of the image file.
        """
        assert type(image_path) == str, "Given path must be a string!"

        img = read_image(image_path, format="BGR")

        return self.get_prediction_from_image(img)


if __name__ == "__main__":
    predictor = Predictor()
    preds = predictor.get_prediction_from_path("0_FRONT.png")

    print(preds)