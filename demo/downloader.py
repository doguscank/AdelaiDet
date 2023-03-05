import json
import os

annot_url = "https://data.votchallenge.net/vot2022/sts/{}.zip"
image_url = "https://data.votchallenge.net/sequences/{}.zip"

with open("description.json", "r") as f:
    data = json.load(f)

sequences = data["sequences"]

for sequence in sequences:
    name = sequence["name"]
    
    image_uid = sequence["channels"]["color"]["uid"]

    dataset_path = os.path.join(
        os.path.expanduser("~"), "AdelaiDet", "datasets", "vot2022", name
    )

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    else:
        continue

    url_annotation = annot_url.format(name)
    url_images = image_url.format(image_uid)

    os.system(
        f"cd {dataset_path} && wget {url_annotation} && unzip *.zip -d annotations && rm *.zip"
    )

    os.system(
        f"cd {dataset_path} && wget {url_images} && unzip *.zip -d images && rm *.zip"
    )
