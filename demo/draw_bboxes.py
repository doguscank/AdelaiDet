import cv2
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# Define the class names
class_names = ["other", "vehicle", "human"]

# Define the colors for GT and predicted bounding boxes
colors = {"gt": (0, 255, 0), "pred": (255, 0, 0)}


def read_bboxes(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        values = line.split()
        class_id = int(values[0])
        if len(values) == 6:
            # Eliminate pred score
            values = [values[i] for i in range(6) if i != 1]
        x, y, w, h = [float(x) for x in values[1:]]
        bboxes.append((x, y, w, h, class_id))
    return bboxes


def draw_bboxes(image_path, gt_bboxes, pred_bboxes):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the ground truth bounding boxes
    for bbox in gt_bboxes:
        x, y, w, h, class_id = bbox
        x, y, w, h = (
            int(x * image.shape[1]),
            int(y * image.shape[0]),
            int(w * image.shape[1]),
            int(h * image.shape[0]),
        )
        cv2.rectangle(
            image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), colors["gt"], 2
        )
        cv2.putText(
            image,
            class_names[class_id],
            (x + w // 2, y - h // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colors["gt"],
            2,
        )

    # Draw the predicted bounding boxes
    for bbox in pred_bboxes:
        x, y, w, h, class_id = bbox
        x, y, w, h = (
            int(x * image.shape[1]),
            int(y * image.shape[0]),
            int(w * image.shape[1]),
            int(h * image.shape[0]),
        )
        cv2.rectangle(
            image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), colors["pred"], 2
        )
        cv2.putText(
            image,
            class_names[class_id],
            (x - w // 2, y - h // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colors["pred"],
            2,
        )

    # Save the image
    image_name = os.path.basename(image_path).split(".")[0]
    plt.imsave(f"{os.path.dirname(image_path)}/{image_name}_plotted.png", image)


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

for segment in tqdm(segments):
    # Define the paths to the image and bounding box directories
    # segment = "segment-15445436653637630344_3957_561_3977_561_with_camera_labels"
    image_dir = f"/home/dogus/final_ws/solov2_venv/src/AdelaiDet/datasets/waymo/{segment}/images"
    gt_bbox_dir = f"/home/dogus/waymo_metrics/{segment}/gts_no_other"
    pred_bbox_dir = f"/home/dogus/waymo_metrics/{segment}/preds_no_other"

    # Get the list of image files
    image_files = glob.glob(os.path.join(image_dir, "*FRONT.png"))

    for idx, image_file in tqdm(list(enumerate(image_files))):
        # if idx % 5 != 0:
        # continue

        # Get the corresponding bounding box files
        base_name = os.path.basename(image_file).split(".")[0]
        gt_bbox_file = os.path.join(gt_bbox_dir, f"{base_name}.txt")
        pred_bbox_file = os.path.join(pred_bbox_dir, f"{base_name}.txt")

        # Read the bounding boxes
        gt_bboxes = read_bboxes(gt_bbox_file)
        pred_bboxes = read_bboxes(pred_bbox_file)

        # Draw the bounding boxes on the image
        draw_bboxes(image_file, gt_bboxes, pred_bboxes)
