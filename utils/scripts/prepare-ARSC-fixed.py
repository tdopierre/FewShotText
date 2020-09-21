import shutil
import os
from utils.data import get_tsv_data

os.makedirs("data/ARSC-fixed", exist_ok=True)

with open("data/ARSC-Yu/raw/workspace.filtered.list", "r") as file:
    train_labels = [line.strip() for line in file.readlines()]
with open("data/ARSC-Yu/raw/workspace.target.list", "r") as file:
    test_labels = [line.strip() for line in file.readlines()]

train_data = list()

for lab in train_labels:
    for ix in (2, 4, 5):
        for split_type in ("train", "dev", "test"):
            train_data += get_tsv_data(f"data/ARSC-Yu/raw/{lab}.t{ix}.{split_type}", label=lab)

test_data = list()

for lab in test_labels:
    for ix in (2, 4, 5):
        for split_type in ("train", "dev", "test"):
            test_data += get_tsv_data(f"data/ARSC-Yu/raw/{lab}.t{ix}.{split_type}", label=lab)
