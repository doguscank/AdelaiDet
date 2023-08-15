import os
import cv2
import numpy as np
import glob
import random
import re
from tqdm import tqdm

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


def random_color():
    return np.array(
        [random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)],
        dtype=np.uint8,
    )


def load_bbox(file):
    with open(file, "r") as f:
        lines = f.readlines()
        coco_categories = [CLASSES[int(line.split(",")[0])] for line in lines]
        for idx, coco_cate in enumerate(coco_categories):
            if coco_cate in COCO_CLASSES["human"]:
                coco_categories[idx] = "human"
            elif coco_cate in COCO_CLASSES["vehicle"]:
                coco_categories[idx] = "vehicle"
            else:
                coco_categories[idx] = "other"

        probabilities = [float(line.split(",")[1]) for line in lines]
        bboxes = [[int(float(x)) for x in line.split(",")[2:6]] for line in lines]
    return coco_categories, probabilities, bboxes


def load_bbox_gt(file):
    with open(file, "r") as f:
        lines = f.readlines()
        coco_categories = [CLASSES[int(line.split()[0])] for line in lines]
        for idx, coco_cate in enumerate(coco_categories):
            if coco_cate in COCO_CLASSES["human"]:
                coco_categories[idx] = "human"
            elif coco_cate in COCO_CLASSES["vehicle"]:
                coco_categories[idx] = "vehicle"
            else:
                coco_categories[idx] = "other"

        bboxes = [[float(x) for x in line.split()[1:5]] for line in lines]
    return coco_categories, bboxes


def put_text(image, mask, category, probability):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_mask)
    center = (
        int(moments["m10"] / (moments["m00"] + 1e-6)),
        int(moments["m01"] / (moments["m00"] + 1e-6)),
    )
    text = f"{category} %{int(probability * 100)}"
    cv2.putText(
        image,
        text,
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def sort_key(file_name):
    return int(file_name.split("_")[-1][:-4])


base_image_files_path = (
    "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/datasets/waymo/{}/images/*FRONT.png"
)

base_bbox_files_path = "/home/dogus/waymo_metrics/{}/preds/{}.txt"
base_bbox_files_path = (
    "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/results/waymo/{}/bboxes/{}.txt"
)

base_mask_files_path = (
    "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/results/waymo/{}/masks/{}"
)
base_gt_bbox_files_path = "/home/dogus/waymo_metrics/{}/gts_no_other/{}.txt"
segment = "segment-15445436653637630344_3957_561_3977_561_with_camera_labels"
image_files = glob.glob(base_image_files_path.format(segment))

segments = [
    "segment-10206293520369375008_2796_800_2816_800_with_camera_labels",
    # "segment-10241508783381919015_2889_360_2909_360_with_camera_labels",
    "segment-10500357041547037089_1474_800_1494_800_with_camera_labels",
    # "segment-10526338824408452410_5714_660_5734_660_with_camera_labels",
    # "segment-10724020115992582208_7660_400_7680_400_with_camera_labels",
    # "segment-11004685739714500220_2300_000_2320_000_with_camera_labels",
    "segment-11119453952284076633_1369_940_1389_940_with_camera_labels",
    # "segment-11355519273066561009_5323_000_5343_000_with_camera_labels",
    # "segment-11623618970700582562_2840_367_2860_367_with_camera_labels",
    # "segment-11799592541704458019_9828_750_9848_750_with_camera_labels",
    "segment-1208303279778032257_1360_000_1380_000_with_camera_labels",
    # "segment-12257951615341726923_2196_690_2216_690_with_camera_labels",
    # "segment-12304907743194762419_1522_000_1542_000_with_camera_labels",
    # "segment-12581809607914381746_1219_547_1239_547_with_camera_labels",
    # "segment-13519445614718437933_4060_000_4080_000_with_camera_labels",
    # "segment-14106113060128637865_1200_000_1220_000_with_camera_labels",
    # "segment-14143054494855609923_4529_100_4549_100_with_camera_labels",
    # "segment-14233522945839943589_100_000_120_000_with_camera_labels",
    # "segment-14753089714893635383_873_600_893_600_with_camera_labels",
    # "segment-15036582848618865396_3752_830_3772_830_with_camera_labels",
    # "segment-15374821596407640257_3388_480_3408_480_with_camera_labels",
    # "segment-15445436653637630344_3957_561_3977_561_with_camera_labels",
    # "segment-15533468984793020049_800_000_820_000_with_camera_labels",
    # "segment-15578655130939579324_620_000_640_000_with_camera_labels",
    # "segment-15857303257471811288_1840_000_1860_000_with_camera_labels",
]

for segment in segments:
    os.makedirs(f"/home/dogus/waymo_maskeds/{segment}", exist_ok=True)
    os.makedirs(f"/home/dogus/waymo_boxeds/{segment}", exist_ok=True)

    image_files = glob.glob(base_image_files_path.format(segment))

    for idx, image_file in tqdm(list(enumerate(image_files))):
        if idx % 20 != 0:
            continue
        image = cv2.imread(image_file)
        bbox_image = image.copy()

        base_name = os.path.basename(image_file).split(".")[0]
        mask_files = sorted(
            glob.glob(f"{base_mask_files_path.format(segment, base_name)}/*.png"),
            key=sort_key,
        )
        bbox_file = base_bbox_files_path.format(segment, base_name)
        gt_bbox_file = base_gt_bbox_files_path.format(segment, base_name)
        total_mask = np.zeros((1280, 1920, 3), dtype=np.uint8)

        if not os.path.exists(bbox_file):
            continue

        categories, probabilities, bboxes = load_bbox(bbox_file)
        gt_categories, gt_bboxes = load_bbox_gt(gt_bbox_file)

        for mask_file, category, probability in zip(
            mask_files, categories, probabilities
        ):
            if category != "other":
                mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
                color = random_color()

                mask_indices = np.where(mask > 0)
                mask[mask_indices[0], mask_indices[1], 0] = color[0]
                mask[mask_indices[0], mask_indices[1], 1] = color[1]
                mask[mask_indices[0], mask_indices[1], 2] = color[2]

                total_mask[mask_indices] = mask[mask_indices]
                put_text(image, mask, category, probability)

        image = cv2.addWeighted(image, 1, total_mask, 0.7, 0)
        cv2.imwrite(f"/home/dogus/waymo_maskeds/{segment}/{base_name}.png", image)

        for bbox, category, probability in zip(bboxes, categories, probabilities):
            if category != "other":
                x1, y1, x2, y2 = bbox
                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    bbox_image,
                    f"{category}({probability:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        for cate, bbox in zip(gt_categories, gt_bboxes):
            x, y, w, h = bbox
            x, y, w, h = int(x * 1920), int(y * 1280), int(w * 1920), int(h * 1280)
            cv2.rectangle(
                bbox_image,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                bbox_image,
                cate,
                (x - w // 2, y + h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(f"/home/dogus/waymo_boxeds/{segment}/{base_name}.png", bbox_image)
