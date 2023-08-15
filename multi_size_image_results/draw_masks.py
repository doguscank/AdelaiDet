import os
import cv2
import numpy as np
import glob
import random
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

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
    "TYPE_CYCLIST": "human"
}

NEW_IDS = {
    "other": 0,
    "vehicle": 1,
    "human": 2,
}

PADDING = 10  # adjust this value as needed

import math

FIG_SIZE = (15, 15)  # adjust this as needed
DPI = 300  # adjust this as needed

def crop_image(image, bbox, padding=PADDING):
    x1, y1, x2, y2 = bbox
    h, w, _ = image.shape
    y1 = max(0, y1 - padding)
    x1 = max(0, x1 - padding)
    y2 = min(h, y2 + padding)
    x2 = min(w, x2 + padding)
    return image[y1:y2, x1:x2]

def random_color():
    return np.array([random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)], dtype=np.uint8)

def load_bbox(file):
    with open(file, 'r') as f:
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
    with open(file, 'r') as f:
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
    center = (int(moments['m10'] / (moments['m00'] + 1e-6)), int(moments['m01'] / (moments['m00'] + 1e-6)))
    text = f"{category} %{int(probability * 100)}"
    cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)

def sort_key(file_name):
    return int(file_name.split("_")[-1][:-4])

def remove_empty_subplots(fig):
    for ax in fig.axes:
        if not any([artist for artist in ax.get_children() if isinstance(artist, matplotlib.lines.Line2D)]):
            ax.remove()

base_image_files_path = "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/multi_size_image/*.png"
base_bbox_files_path = "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/multi_size_image_results/bboxes/{}.txt"
base_mask_files_path = "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/multi_size_image_results/masks/{}"

os.makedirs("/home/dogus/multisizemasked", exist_ok=True)
os.makedirs("/home/dogus/multisizebbox", exist_ok=True)
os.makedirs("/home/dogus/multisizefigs", exist_ok=True)

image_files = glob.glob(base_image_files_path)

# for idx, image_file in tqdm(list(enumerate(image_files))):
#     if not "cropped" in image_file:
#         continue
    
#     image = cv2.imread(image_file)
#     bbox_image = image.copy()
    
#     base_name = os.path.basename(image_file).split('.')[0]
#     mask_files = sorted(glob.glob(f"{base_mask_files_path.format(base_name)}/*.png"), key=sort_key)
#     bbox_file = base_bbox_files_path.format(base_name)
#     total_mask = np.zeros_like(image, dtype=np.uint8)
    
#     if not os.path.exists(bbox_file):
#         continue
    
#     categories, probabilities, bboxes = load_bbox(bbox_file)

#     mask_count = len(categories) #sum(1 for category in categories if category != "other")
#     rows = math.ceil(mask_count / 5)
#     figure, axes = plt.subplots(rows, 5, figsize=FIG_SIZE, dpi=DPI)
#     axes = axes.ravel()  # flatten the axes array

#     for i, (mask_file, category, probability, bbox) in enumerate(zip(mask_files, categories, probabilities, bboxes)):
#         if category != "other":
#             mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
#             color = random_color()
            
#             mask_indices = np.where(mask > 0)
#             mask[mask_indices[0], mask_indices[1], 0] = color[0]
#             mask[mask_indices[0], mask_indices[1], 1] = color[1]
#             mask[mask_indices[0], mask_indices[1], 2] = color[2]

#             total_mask[mask_indices] = mask[mask_indices]
            
#             instance_image = crop_image(image, bbox)
#             instance_mask = crop_image(mask, bbox)
            
#             instance_masked_image = cv2.bitwise_and(instance_image, instance_mask)
#             instance_masked_image = cv2.addWeighted(instance_image, 0.5, instance_mask, 0.5, 0)
            
#             axes[i].imshow(cv2.cvtColor(instance_masked_image, cv2.COLOR_BGR2RGB))
#             axes[i].set_title(f"{category} %{int(probability * 100)}")
#             axes[i].axis('off')


#     for ax in figure.axes:
#         if ax.get_title() == '':
#             figure.delaxes(ax)


#     plt.tight_layout()
#     plt.savefig(f"/home/dogus/multisizefigs/{base_name}.png", dpi=DPI)
#     plt.cla(); plt.clf()


for idx, image_file in tqdm(list(enumerate(image_files))):
    image = cv2.imread(image_file)
    bbox_image = image.copy()
    
    base_name = os.path.basename(image_file).split('.')[0]
    mask_files = sorted(glob.glob(f"{base_mask_files_path.format(base_name)}/*.png"), key=sort_key)
    bbox_file = base_bbox_files_path.format(base_name)
    total_mask = np.zeros_like(image, dtype=np.uint8)
    
    if not os.path.exists(bbox_file):
        continue
    
    categories, probabilities, bboxes = load_bbox(bbox_file)
    
    for mask_file, category, probability in zip(mask_files, categories, probabilities):
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
    cv2.imwrite(f"/home/dogus/multisizemasked/{base_name}.png", image)
    
    for bbox, category, probability in zip(bboxes, categories, probabilities):
        if category != "other":
            x1, y1, x2, y2 = bbox
            cv2.rectangle(bbox_image, (x1, y1), (x2,y2), (0,0,255), 2)
            cv2.putText(bbox_image, f"{category}({probability:.2f})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
    cv2.imwrite(f"/home/dogus/multisizebbox/{base_name}.png", bbox_image)
        