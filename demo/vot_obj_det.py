import json
import os

import cv2
import numpy as np
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

human_cls = ["person"]
vehicle_cls = ["bicycle", "car", "motorcycle", "bus", "truck"]
animal_cls = ["bird", "cat", "dog", "horse", "sheep", "cow"]
other_cls = list(set(CLASSES) - set(human_cls) - set(vehicle_cls) - set(animal_cls))

human_datasets = [
    "basketball",
    "bolt1",
    "diver",
    "girl",
    "graduate",
    "gymnastics1",
    "gymnastics2",
    "gymnastics3",
    "handball1",
    "handball2",
    "iceskater1",
    "iceskater2",
    "marathon",
    "matrix",
    "polo",
    "rowing",
    "shaking",
    "singer2",
    "singer3",
    "soccer1",
    "soccer2",
    "soldier",
    "surfing",
]
vehicle_datasets = ["car1", "wiper"]
animal_datasets = [
    "agility",
    "animal",
    "ants1",
    "birds1",
    "birds2",
    "butterfly",
    "crabs1",
    "fernando",
    "fish1",
    "fish2",
    "flamingo1",
    "kangaroo",
    "lamb",
    "monkey",
    "nature",
    "rabbit",
    "rabbit2",
    "snake",
    "tiger",
    "zebrafish1",
]
other_datasets = [
    "bag",
    "ball2",
    "ball3",
    "book",
    "bubble",
    "conduction",
    "dinosaur",
    "drone1",
    "drone_across",
    "frisbee",
    "hand",
    "hand2",
    "helicopter",
    "leaves",
    "motocross1",
    "tennis",
    "wheel",
]

cls_dict = {"human": 1, "vehicle": 2, "animal": 3, "other": 4}


def most_frequent(List):
    if len(List) == 0:
        return ""
    return max(set(List), key=List.count)


def rle_to_mask(rle, shape):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    width, height = shape
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_ + j] = 255
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def IoU(pred_mask, true_mask):
    assert (
        pred_mask.shape == true_mask.shape
    ), f"Masks have different shapes: pred: {pred_mask.shape} gt: {true_mask.shape}"
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def find_bounding_box(mask, shape):
    # The input mask is a binary image (numpy array), so we find all the non-zero points
    points = cv2.findNonZero(mask)

    # Then we use the OpenCV function boundingRect to get the bounding box around these points
    x, y, w, h = cv2.boundingRect(points)

    # The function returns the top-left corner coordinates and the width and height of the box
    return x / shape[0], y / shape[1], w / shape[0], h / shape[1]


def dataset_evaluator(vot22_dataset_name, min_iou=0.5):
    gt_path = os.path.join(
        "datasets", "vot2022", vot22_dataset_name, "annotations", "groundtruth.txt"
    )

    image_dir = os.path.join("datasets", "vot2022", vot22_dataset_name, "images")
    masked_detections_dir = os.path.join(
        "results",
        "vot2022",
        f"{vot22_dataset_name}",
        f"masked_detections_{str(f'{min_iou:.1f}').replace('.', '_')}",
    )
    masks_dir = os.path.join(
        "results", "vot2022", f"{vot22_dataset_name}", "masks", "{0}"
    )
    gt_dir = os.path.join("results", "vot2022", f"{vot22_dataset_name}", "gt")
    bboxes_dir = os.path.join(
        "results", "vot2022", f"{vot22_dataset_name}", "bboxes", "{0}.txt"
    )

    gt_bboxes_dir = os.path.join(
        os.path.expanduser("~"), "vot", f"{vot22_dataset_name}", "gt"
    )
    pred_bboxes_dir = os.path.join(
        os.path.expanduser("~"), "vot", f"{vot22_dataset_name}", "pred"
    )

    os.makedirs(gt_bboxes_dir, exist_ok=True)
    os.makedirs(pred_bboxes_dir, exist_ok=True)

    shape = None

    if not os.path.exists(f"results/vot2022/{vot22_dataset_name}"):
        os.mkdir(f"results/vot2022/{vot22_dataset_name}")

    if not os.path.exists(masked_detections_dir):
        os.mkdir(masked_detections_dir)

    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)

    if not os.path.exists(masks_dir[:-4]):
        os.mkdir(masks_dir[:-4])

    image_paths = sorted(os.listdir(image_dir))

    shape = cv2.imread(os.path.join(image_dir, image_paths[0])).shape[:2][::-1]

    with open(gt_path, "r") as f:
        rle_data = f.read().split("\n")[:-1]

        image_list = []

        for idx, rle in tqdm(enumerate(rle_data)):
            # if idx == 10:
            #     exit()
            splitted_rle = rle[1:].split(",")
            bbox = splitted_rle[:4]
            x0, y0, w, h = np.array(bbox, dtype=int)
            mask_rle = np.array(splitted_rle[4:], dtype=int)

            instance_mask = rle_to_mask(mask_rle, (w, h))
            mask = np.zeros(shape[::-1])
            mask[y0 : y0 + h, x0 : x0 + w] = instance_mask

            cv2.imwrite(os.path.join(gt_dir, "{:08}.png".format(idx + 1)), mask)

            image_list.append(cv2.imread(os.path.join(image_dir, image_paths[idx])))

            image_name = image_paths[idx].split(".")[0]

            gt_label = cls_dict["other"]

            if vot22_dataset_name in human_datasets:
                gt_label = cls_dict["human"]
            elif vot22_dataset_name in animal_datasets:
                gt_label = cls_dict["animal"]
            elif vot22_dataset_name in vehicle_datasets:
                gt_label = cls_dict["vehicle"]

            with open(os.path.join(gt_bboxes_dir, f"{image_name}.txt"), "w") as gt_f:
                gt_f.write(
                    f"{gt_label} {x0 / shape[0]} {y0 / shape[1]} {w / shape[0]} {h / shape[1]}"
                )

            try:
                mask_paths = sorted(os.listdir(masks_dir.format(image_name)))
            except:
                print("No mask paths")
                continue

            if len(mask_paths) == 0:
                continue

            max_iou = min_iou  # initial value is threshold value
            max_iou_idx = -1

            for idxx, mask_path in enumerate(mask_paths):
                pred_mask = cv2.imread(
                    os.path.join(masks_dir.format(image_name), mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                iou_score = IoU(pred_mask, mask)

                if iou_score > max_iou:
                    max_iou = iou_score
                    max_iou_idx = idxx

            if max_iou_idx == -1:
                open(os.path.join(pred_bboxes_dir, f"{image_name}.txt"), "w").close()
                continue

            with open(bboxes_dir.format(image_name), "r") as f:
                bbox_data = f.read().split("\n")[:-1]

            mask_class = CLASSES[int(bbox_data[max_iou_idx].split(",")[0])]
            confidence = float(bbox_data[max_iou_idx].split(",")[1])

            if mask_class in human_cls:
                mask_label = cls_dict["human"]
            elif mask_class in animal_cls:
                mask_label = cls_dict["animal"]
            elif mask_class in vehicle_cls:
                mask_label = cls_dict["vehicle"]
            else:
                mask_label = cls_dict["other"]

            pred_mask = cv2.imread(
                os.path.join(masks_dir.format(image_name), mask_paths[max_iou_idx]),
                cv2.IMREAD_GRAYSCALE,
            )

            x, y, w, h = find_bounding_box(pred_mask, shape)

            with open(
                os.path.join(pred_bboxes_dir, f"{image_name}.txt"), "w"
            ) as pred_f:
                pred_f.write(f"{mask_label} {confidence} {x} {y} {w} {h}")


if __name__ == "__main__":
    dataset_evaluator("zebrafish1", min_iou=0.5)
    exit()

    vot_datasets = [*human_datasets, *vehicle_datasets, *animal_datasets]

    for dataset in vot_datasets:
        dataset_evaluator(dataset, min_iou=0.5)
        # exit()
