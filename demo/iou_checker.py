import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from adet.modeling.MEInst.LME import IOUMetric

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

human_datasets = ["basketball", "bolt1", "diver", "girl", "graduate", "gymnastics1", "gymnastics2", "gymnastics3", "handball1", "handball2", "iceskater1", "iceskater2", "marathon", "matrix", "polo", "rowing", "shaking", "singer2", "singer3", "soccer1", "soccer2", "soldier", "surfing"]
vehicle_datasets = ["car1", "wiper"]
animal_datasets = ["agility", "animal", "ants1", "birds1", "birds2", "butterfly", "crabs1", "fernando", "fish1", "fish2", "flamingo1", "kangaroo", "lamb", "monkey", "nature", "rabbit", "rabbit2", "snake", "tiger", "zebrafish"]
other_datasets = ["bag", "ball2", "ball3", "book", "bubble", "conduction", "dinosaur", "drone1", "drone_across", "frisbee", "hand", "hand2", "helicopter", "leaves", "motocross1", "tennis", "wheel"]

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


def pixel_accuracy(pred_mask, true_mask):
    assert pred_mask.shape == true_mask.shape, "Masks have different shapes"
    correct_pixels = np.sum(pred_mask == true_mask)
    total_pixels = np.prod(pred_mask.shape)
    acc = correct_pixels / total_pixels
    return acc


def dice_coeff(pred_mask, true_mask):
    assert pred_mask.shape == true_mask.shape, "Masks have different shapes"
    dice = (
        np.sum(pred_mask[true_mask == 255])
        / 255
        * 2.0
        / (np.sum((pred_mask / 255) ** 2) + np.sum((true_mask / 255) ** 2))
    )
    return dice


def create_masked_video(images, masks, output_file, fps, dataset_name, min_iou):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = images[0].shape
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    masked_detections_dir = os.path.join(
        "results", f"{dataset_name}", "masked_detections_" + (f'{min_iou:.1f}').replace('.', '_')
    )

    for i in range(len(images)):
        img = images[i]
        mask = masks[i]

        # color the mask red
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask[:, :, 0] = 0
        mask[:, :, 1] = 0

        # add the colored mask to the image with 0.7 alpha value
        try:
            img = cv2.addWeighted(src1=img, alpha=0.7, src2=mask, beta=0.3, gamma=0)
        except:
            print(img.shape, mask.shape, img.dtype, mask.dtype)

        cv2.imwrite(os.path.join(masked_detections_dir, f"{(i+1):08}.png"), img)
        out.write(img)

    out.release()


def dataset_evaluator(vot22_dataset_name, min_iou=0.5):
    gt_path = os.path.join(
        "datasets", "vot2022", vot22_dataset_name, "annotations", "groundtruth.txt"
    )

    image_dir = os.path.join("datasets", "vot2022", vot22_dataset_name, "images")
    masked_detections_dir = os.path.join(
        "results", f"{vot22_dataset_name}", f"masked_detections_{str(f'{min_iou:.1f}').replace('.', '_')}"
    )
    masks_dir = os.path.join("results", f"{vot22_dataset_name}", "masks", "{0}")
    gt_dir = os.path.join("results", f"{vot22_dataset_name}", "gt")
    bboxes_dir = os.path.join("results", f"{vot22_dataset_name}", "bboxes", "{0}.txt")

    shape = None

    if not os.path.exists(f"results/{vot22_dataset_name}"):
        os.mkdir(f"results/{vot22_dataset_name}")

    if not os.path.exists(masked_detections_dir):
        os.mkdir(masked_detections_dir)

    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)

    if not os.path.exists(masks_dir[:-4]):
        os.mkdir(masks_dir[:-4])

    image_paths = sorted(os.listdir(image_dir))
    iou_values = []
    accuracy_values = []
    dice_values = []

    mask_classes = {}
    mask_class_list = []

    shape = cv2.imread(os.path.join(image_dir, image_paths[0])).shape[:2][::-1]

    with open(gt_path, "r") as f:
        rle_data = f.read().split("\n")[:-1]

        image_list = []
        mask_list = []

        open(os.path.join("results", f"{vot22_dataset_name}", "ious.txt"), "w").close()

        print(vot22_dataset_name)
        for idx, rle in tqdm(enumerate(rle_data)):
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
            try:
                mask_paths = sorted(os.listdir(masks_dir.format(image_name)))
            except:
                print("No mask paths")
                iou_values.append(0)
                accuracy_values.append(0)
                dice_values.append(0)
                mask_list.append(np.zeros(shape[::-1], dtype=np.uint8))
                continue

            if len(mask_paths) == 0:
                iou_values.append(0)
                accuracy_values.append(0)
                dice_values.append(0)
                mask_list.append(np.zeros(shape[::-1], dtype=np.uint8))
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
                iou_values.append(0)
                accuracy_values.append(0)
                dice_values.append(0)
                mask_list.append(np.zeros(shape[::-1], dtype=np.uint8))
                continue

            with open(bboxes_dir.format(image_name), "r") as f:
                bbox_data = f.read().split("\n")[:-1]

            mask_class = CLASSES[int(bbox_data[max_iou_idx].split(",")[0])]

            for dataset, dataset_cls in zip([human_datasets, vehicle_datasets, animal_datasets], [human_cls, vehicle_cls, animal_cls]):
                if vot22_dataset_name in dataset:
                    if mask_class in dataset_cls:
                        break
                elif vot22_dataset_name in other_datasets:
                    if mask_class in other_cls:
                        break
            else:
                iou_values.append(0)
                accuracy_values.append(0)
                dice_values.append(0)
                mask_list.append(np.zeros(shape[::-1], dtype=np.uint8))
                continue
            

            mask_classes[idx] = mask_class
            mask_class_list.append(mask_class)

            with open(
                os.path.join("results", f"{vot22_dataset_name}", "ious.txt"), "a"
            ) as f:
                f.write(f"name:{(idx + 1):08},iou:{max_iou:.3f},class:{CLASSES[int(bbox_data[max_iou_idx].split(',')[0])]}\n")

            mask_list.append(
                cv2.imread(
                    os.path.join(masks_dir.format(image_name), mask_paths[max_iou_idx]),
                    cv2.IMREAD_GRAYSCALE,
                )
            )

            iou_values.append(max_iou)
            accuracy_values.append(
                pixel_accuracy(
                    cv2.imread(
                        os.path.join(
                            masks_dir.format(image_name), mask_paths[max_iou_idx]
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    ),
                    mask,
                )
            )
            dice_values.append(
                dice_coeff(
                    cv2.imread(
                        os.path.join(
                            masks_dir.format(image_name), mask_paths[max_iou_idx]
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    ),
                    mask,
                )
            )

    iou_values = np.array(iou_values)
    accuracy_values = np.array(accuracy_values)
    dice_values = np.array(dice_values)

    miou = np.mean(iou_values)
    macc = np.mean(accuracy_values)
    mdice = np.mean(dice_values)

    # if miou != 0.0:
    #     nonzero_miou = np.mean(iou_values[iou_values != 0.0])
    # else:
    #     nonzero_miou = 0.0

    # if macc != 0.0:
    #     nonzero_macc = np.mean(accuracy_values[accuracy_values != 0.0])
    # else:
    #     nonzero_macc = 0.0

    # if mdice != 0.0:
    #     nonzero_mdice = np.mean(dice_values[dice_values != 0.0])
    # else:
    #     nonzero_mdice = 0.0

    find_rates_n = np.sum(np.array([iou_values != 0.0], dtype=int))
    find_rates_d = len(iou_values)
    find_rates = find_rates_n / find_rates_d

    print("miou:", miou)
    print("acc:", macc)
    print("dice:", mdice)
    print("find_rates:", find_rates)

    fps = 30
    # create_masked_video(
    #     image_list, mask_list, os.path.join("results", vot22_dataset_name, f"_{str(f'{min_iou:.1f}').replace('.', '_')}.mp4"), fps, vot22_dataset_name, min_iou
    # )
    with open(f"results_{str(f'{min_iou:.1f}').replace('.', '_')}.csv", "a") as f:
        f.write(
            f"{vot22_dataset_name},{find_rates_d},{find_rates_n},{find_rates:.3f},{macc:.3f},{miou:.3f},{mdice:.3f},{most_frequent(mask_class_list)},{mask_class_list.count(most_frequent(mask_class_list))}\n"
        )

    # with open("results_nz.txt", "a") as f:
    #     f.write(
    #         f"{vot22_dataset_name},{nonzero_miou:.3f},{nonzero_macc:.3f},{nonzero_mdice:.3f},{find_rates_n:.3f},{find_rates_d:.3f},{find_rates:.3f},{most_frequent(mask_class_list)},{mask_class_list.count(most_frequent(mask_class_list))}\n"
    #     )


if __name__ == "__main__":
    with open("description.json", "r") as f:
        data = json.load(f)

    vot_datasets = []
    iou_values_to_test = [0.1 * i for i in range(10)]

    iou_values_to_test = iou_values_to_test[1:]

    for seq in data["sequences"]:
        vot_datasets.append(seq["name"])

    # vot_datasets = ["bolt1"]

    for dataset in vot_datasets:
        os.makedirs("results/masks", 0o755, exist_ok=True)
        
        if os.path.exists(f"results_{dataset}/"):
            os.system(f"mv results_{dataset} results/{dataset}")
            continue

        if os.path.exists(f"results/{dataset}/"):
            print(f"results/{dataset} exists. Continuing...")
            continue

        os.system(
            f"python demo/demo.py --config-file configs/SOLOv2/R101_3x.yaml --input ./datasets/vot2022/{dataset}/images --output ./results/{dataset} --opts MODEL.WEIGHTS weights/SOLOv2_R101_3x.pth"
        )
        os.system(f"mv results/masks results/{dataset}/masks")

    for iou in iou_values_to_test:
        with open(f"results_{f'{iou:.1f}'.replace('.', '_')}.csv", "w") as f:
            f.write(
                f"dataset,total_images,found,find_rates,acc,miou,dice,most_repetitive_class,most_repetitive_class_count\n"
            )

    # with open("results_nz.txt", "w") as f:
    #     f.write(
    #         f"dataset,miou,acc,dice,found,total_images,find_rates,most_repetitive_class,most_repetitive_class_count\n"
    #     )

    for iou in iou_values_to_test:
        for dataset in vot_datasets:
            dataset_evaluator(dataset, min_iou=iou)
