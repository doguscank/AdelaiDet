import os

import cv2
import numpy as np
from tqdm import tqdm

from adet.modeling.MEInst.LME import IOUMetric

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
                v[idx_+j] = 255
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))

def IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def pixel_accuracy(pred_mask, true_mask):
    assert pred_mask.shape == true_mask.shape, "Masks have different shapes"
    correct_pixels = np.sum(pred_mask == true_mask)
    total_pixels = np.prod(pred_mask.shape)
    acc = correct_pixels / total_pixels
    return acc

def create_masked_video(images, masks, output_file, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = images[0].shape
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i in range(len(images)):
        img = images[i]
        mask = masks[i]

        # color the mask red
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask[:,:,0] = 0
        mask[:,:,1] = 0
        
        # add the colored mask to the image with 0.7 alpha value
        try:
            img = cv2.addWeighted(src1=img, alpha=0.7, src2=mask, beta=0.3, gamma=0)
        except:
            print(img.shape, mask.shape, img.dtype, mask.dtype)
        
        out.write(img)

    out.release()


if __name__ == "__main__":
    vot22_dataset_name = "car1"
    shape = (640, 480)

    gt_path = os.path.join(
        "datasets",
        "vot2022",
        vot22_dataset_name,
        "annotations",
        "groundtruth.txt"
    )

    image_dir = os.path.join(
        "datasets",
        "vot2022",
        vot22_dataset_name,
        "images"
    )

    masks_dir = os.path.join(
        f"results_{vot22_dataset_name}",
        "masks",
        "{0}"
    )

    image_paths = sorted(os.listdir(image_dir))
    iou_values = []
    accuracy_values = []

    with open(gt_path, "r") as f:
        rle_data = f.read().split("\n")[:-1]

        image_list = []
        mask_list = []

        for idx, rle in tqdm(enumerate(rle_data)):
            splitted_rle = rle[1:].split(",")
            bbox = splitted_rle[:4]
            x0, y0, w, h = np.array(bbox, dtype=int)
            mask_rle = np.array(splitted_rle[4:], dtype=int)

            instance_mask = rle_to_mask(mask_rle, (w, h))
            mask = np.zeros(shape[::-1])
            mask[y0: y0+h, x0: x0+w] = instance_mask

            image_list.append(
                cv2.imread(os.path.join(
                    image_dir,
                    image_paths[idx]
                ))
            )
            
            image_name = image_paths[idx].split(".")[0]
            mask_paths = sorted(os.listdir(masks_dir.format(image_name)))

            if len(mask_paths) == 0:
                iou_values.append(0)
                accuracy_values.append(0)
                mask_list.append(np.zeros(shape[::-1], dtype=np.uint8))
                continue

            max_iou = 0
            max_iou_idx = 0
            
            for idxx, mask_path in enumerate(mask_paths):
                pred_mask = cv2.imread(
                    os.path.join(
                        masks_dir.format(image_name),
                        mask_path
                    ),
                    cv2.IMREAD_GRAYSCALE
                )

                iou_score = IoU(pred_mask, mask)
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_iou_idx = idxx

            mask_list.append(
                cv2.imread(
                    os.path.join(
                        masks_dir.format(image_name),
                        mask_paths[max_iou_idx]
                    ),
                    cv2.IMREAD_GRAYSCALE
                )
            )

            iou_values.append(max_iou)
            accuracy_values.append(pixel_accuracy(
                cv2.imread(
                    os.path.join(
                        masks_dir.format(image_name),
                        mask_paths[max_iou_idx]
                    ),
                    cv2.IMREAD_GRAYSCALE
                ),
                mask
            ))

    miou = sum(iou_values) / len(iou_values)
    acc = sum(accuracy_values) / len(accuracy_values)
    print("miou:", miou)
    print("acc:", acc)

    fps = 30
    create_masked_video(image_list, mask_list, vot22_dataset_name + ".mp4", fps)
