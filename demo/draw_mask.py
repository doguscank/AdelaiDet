import os
import cv2
import numpy as np
import glob
import random
import re
from tqdm import tqdm

def random_color():
    return np.array([random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)], dtype=np.uint8)

def load_bbox(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        categories = [line.split()[0] for line in lines]
        probabilities = [float(line.split()[1]) for line in lines]
    return categories, probabilities

def put_text(image, mask, category, probability):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_mask)
    center = (int(moments['m10'] / (moments['m00'] + 1e-6)), int(moments['m01'] / (moments['m00'] + 1e-6)))
    text = f"{category} %{int(probability * 100)}"
    cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def sort_key(file_name):
    return int(file_name.split("_")[-1][:-4])

base_image_files_path = "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/datasets/waymo/{}/images/*FRONT.png"
base_bbox_files_path = "/home/dogus/waymo_metrics/{}/preds/{}.txt"
base_mask_files_path = "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/results/waymo/{}/masks/{}"
segment = "segment-15445436653637630344_3957_561_3977_561_with_camera_labels"
image_files = glob.glob(base_image_files_path.format(segment))
desired_categories = ["1", "2"]
category_names = {
    "1": "vehicle",
    "2": "human"
}

segments = [
    "segment-10206293520369375008_2796_800_2816_800_with_camera_labels",
    "segment-10241508783381919015_2889_360_2909_360_with_camera_labels",
    "segment-10500357041547037089_1474_800_1494_800_with_camera_labels",
    "segment-10526338824408452410_5714_660_5734_660_with_camera_labels",
    "segment-10724020115992582208_7660_400_7680_400_with_camera_labels",
    "segment-11004685739714500220_2300_000_2320_000_with_camera_labels",
    "segment-11119453952284076633_1369_940_1389_940_with_camera_labels",
    "segment-11355519273066561009_5323_000_5343_000_with_camera_labels",
    "segment-11623618970700582562_2840_367_2860_367_with_camera_labels",
    "segment-11799592541704458019_9828_750_9848_750_with_camera_labels",
    "segment-1208303279778032257_1360_000_1380_000_with_camera_labels",
    "segment-12257951615341726923_2196_690_2216_690_with_camera_labels",
    "segment-12304907743194762419_1522_000_1542_000_with_camera_labels",
    "segment-12581809607914381746_1219_547_1239_547_with_camera_labels",
    "segment-13519445614718437933_4060_000_4080_000_with_camera_labels",
    "segment-14106113060128637865_1200_000_1220_000_with_camera_labels",
    "segment-14143054494855609923_4529_100_4549_100_with_camera_labels",
    "segment-14233522945839943589_100_000_120_000_with_camera_labels",
    "segment-14753089714893635383_873_600_893_600_with_camera_labels",
    "segment-15036582848618865396_3752_830_3772_830_with_camera_labels",
    "segment-15374821596407640257_3388_480_3408_480_with_camera_labels",
    "segment-15445436653637630344_3957_561_3977_561_with_camera_labels",
    "segment-15533468984793020049_800_000_820_000_with_camera_labels",
    "segment-15578655130939579324_620_000_640_000_with_camera_labels",
    "segment-15857303257471811288_1840_000_1860_000_with_camera_labels",
]

for segment in segments:
    os.makedirs(f"/home/dogus/waymo_maskeds/{segment}", exist_ok=True)
    image_files = glob.glob(base_image_files_path.format(segment))
    
    for idx, image_file in tqdm(list(enumerate(image_files))):
        if idx % 5 != 0:
            continue
        image = cv2.imread(image_file)
        base_name = os.path.basename(image_file).split('.')[0]
        mask_files = sorted(glob.glob(f"{base_mask_files_path.format(segment, base_name)}/*.png"), key=sort_key)
        bbox_file = base_bbox_files_path.format(segment, base_name)
        total_mask = np.zeros((1280, 1920, 3), dtype=np.uint8)
        
        if not os.path.exists(bbox_file):
            continue
        
        categories, probabilities = load_bbox(bbox_file)
        for mask_file, category, probability in zip(mask_files, categories, probabilities):
            if category in desired_categories:
                mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
                color = random_color()
                
                mask_indices = np.where(mask > 0)
                mask[mask_indices[0], mask_indices[1], 0] = color[0]
                mask[mask_indices[0], mask_indices[1], 1] = color[1]
                mask[mask_indices[0], mask_indices[1], 2] = color[2]

                # image = cv2.addWeighted(image, 1, mask, 0.7, 0)
                total_mask[mask_indices] = mask[mask_indices]
                put_text(image, mask, category_names[category], probability)
        
        image = cv2.addWeighted(image, 1, total_mask, 0.7, 0)
        cv2.imwrite(f"/home/dogus/waymo_maskeds/{segment}/{base_name}.png", image)
