import os
import tqdm

folders = [
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

for folder in tqdm.tqdm(folders):
    os.makedirs("results/masks", 0o755, exist_ok=True)

    if os.path.exists(f"results_{folder}/"):
        os.system(f"mv results_{folder} results/waymo/{folder}")
        continue

    if os.path.exists(f"results/waymo/{folder}/"):
        print(f"results/waymo/{folder} exists. Continuing...")
        continue

    os.system(
        f"python demo/demo.py --config-file configs/SOLOv2/R101_3x.yaml --input ./datasets/waymo/{folder}/images --output ./results/waymo/{folder} --opts MODEL.WEIGHTS weights/SOLOv2_R101_3x.pth"
    )
    os.system(f"mv results/masks results/waymo/{folder}/masks")
