from glob import glob
import pandas as pd
import os

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

COCO_CLASSES = {
    "human": ["person"],
    "vehicle": ["bicycle", "car", "motorcycle", "bus", "truck"],
}

WAYMO_CLASSES = {
    "TYPE_VEHICLE": "vehicle",
    "TYPE_PEDESTRIAN": "human",
    "TYPE_CYCLIST": "human",
}

NEW_IDS = {
    "other": 0,
    "vehicle": 1,
    "human": 2,
}

DROP_OTHERS = True


def drop_others(df):
    return df[df["class_label"] != 0]


def pred_cls_to_yolo(cls):
    for k, v in COCO_CLASSES.items():
        if CLASSES[cls] in v:
            return NEW_IDS[k]
    return 0


def waymo_cls_to_yolo(cls):
    return NEW_IDS[WAYMO_CLASSES[cls]]


def pred_to_yolo(df, image_width=1920, image_height=1280):
    """
    This function takes a Pandas DataFrame in format ("class", "confidence", "x1", "y1", "x2", "y2")
    and converts it to YOLO format.
    """
    x_center = (df.iloc[:, 2] + df.iloc[:, 4]) / (2 * image_width)
    y_center = (df.iloc[:, 3] + df.iloc[:, 5]) / (2 * image_height)
    box_width = (df.iloc[:, 4] - df.iloc[:, 2]) / image_width
    box_height = (df.iloc[:, 5] - df.iloc[:, 3]) / image_height
    yolo_box = pd.DataFrame(
        {
            "class_label": df.iloc[:, 0].apply(pred_cls_to_yolo),
            "confidence": df.iloc[:, 1],
            "x_center": x_center,
            "y_center": y_center,
            "box_width": box_width,
            "box_height": box_height,
        }
    )

    if DROP_OTHERS:
        yolo_box = drop_others(yolo_box)

    return yolo_box


def waymo_to_yolo(df, image_width=1920, image_height=1280):
    """
    This function takes a Pandas DataFrame in format ("class", "x1", "y1", "width", "height", "instance_id")
    and removes instance IDs and converts to YOLO format.
    """
    # Calculate ratios to convert bounding box dimensions to YOLO format
    width_ratio = 1.0 / image_width
    height_ratio = 1.0 / image_height

    # Calculate center coordinates and dimensions in YOLO format
    x_center = (df.iloc[:, 1] + (df.iloc[:, 3] / 2)) * width_ratio
    y_center = (df.iloc[:, 2] + (df.iloc[:, 4] / 2)) * height_ratio
    box_width = df.iloc[:, 3] * width_ratio
    box_height = df.iloc[:, 4] * height_ratio

    # Create a DataFrame in YOLO format
    yolo_box = pd.DataFrame(
        {
            "class_label": df.iloc[:, 0].apply(waymo_cls_to_yolo),
            "x_center": x_center,
            "y_center": y_center,
            "box_width": box_width,
            "box_height": box_height,
        }
    )

    if DROP_OTHERS:
        yolo_box = drop_others(yolo_box)

    return yolo_box


if __name__ == "__main__":
    for folder in os.listdir(
        "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/results/waymo"
    ):
        for waymo in sorted(
            glob(
                f"/home/dogus/final_ws/solov2_venv/src/AdelaiDet/datasets/waymo/{folder}/annotations/*.txt"
            )
        ):
            try:
                fn = waymo.split("/")[-1]
                csv_file = pd.read_csv(waymo, header=None)
                df = waymo_to_yolo(csv_file)
                if DROP_OTHERS:
                    os.makedirs(
                        os.path.join(
                            os.path.expanduser("~"),
                            "waymo_metrics",
                            folder,
                            "gts_no_other",
                        ),
                        exist_ok=True,
                    )
                    df.to_csv(
                        os.path.join(
                            os.path.expanduser("~"),
                            "waymo_metrics",
                            folder,
                            "gts_no_other",
                            fn,
                        ),
                        header=None,
                        index=None,
                        sep=" ",
                    )
                else:
                    os.makedirs(
                        os.path.join(
                            os.path.expanduser("~"), "waymo_metrics", folder, "gts"
                        ),
                        exist_ok=True,
                    )
                    df.to_csv(
                        os.path.join(
                            os.path.expanduser("~"), "waymo_metrics", folder, "gts", fn
                        ),
                        header=None,
                        index=None,
                        sep=" ",
                    )
            except:
                print("EXCEPTION")
                print(waymo)
                exit()

        for pred in sorted(
            glob(
                f"/home/dogus/final_ws/solov2_venv/src/AdelaiDet/results/waymo/{folder}/bboxes/*.txt"
            )
        ):
            fn = pred.split("/")[-1]
            df = pred_to_yolo(pd.read_csv(pred, header=None))
            if DROP_OTHERS:
                os.makedirs(
                    os.path.join(
                        os.path.expanduser("~"),
                        "waymo_metrics",
                        folder,
                        "preds_no_other",
                    ),
                    exist_ok=True,
                )
                df.to_csv(
                    os.path.join(
                        os.path.expanduser("~"),
                        "waymo_metrics",
                        folder,
                        "preds_no_other",
                        fn,
                    ),
                    header=None,
                    index=None,
                    sep=" ",
                )
            else:
                os.makedirs(
                    os.path.join(
                        os.path.expanduser("~"), "waymo_metrics", folder, "preds"
                    ),
                    exist_ok=True,
                )
                df.to_csv(
                    os.path.join(
                        os.path.expanduser("~"), "waymo_metrics", folder, "preds", fn
                    ),
                    header=None,
                    index=None,
                    sep=" ",
                )
