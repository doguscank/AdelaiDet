import glob
import os

files = glob.glob("dataset/**/**/*.txt")

for file in files:
    substr = file.split("/")

    os.system(
        f"cp {file} {os.path.join('preds', substr[1], substr[-1])}"
    )