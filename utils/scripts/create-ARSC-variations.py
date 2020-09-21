import random

from utils.data import get_tsv_data
import shutil
import collections

target_labels = [line.strip() for line in open('data/ARSC-Yu/raw/workspace.target.list', 'r').readlines()]

for split_ix in range(1, 6):
    shutil.copytree(src="data/ARSC-Yu/raw", dst=f"data/ARSC-Yu/split{split_ix}")
    for label in target_labels:
        for i in [2, 4, 5]:
            # Load
            data = [line.strip().split('\t') for set_type in ("train", "dev", "test") for line in open(f'data/ARSC-Yu/raw/{label}.t{i}.{set_type}')]

            # Shuffle
            data_dict = collections.defaultdict(list)
            for d in data:
                data_dict[d[1]].append(d)
            for k, v in data_dict.items():
                random.shuffle(v)

            # Split
            train_data = [item for k, v in data_dict.items() for item in v[:5]]
            dev_data = [item for k, v in data_dict.items() for item in v[5:5 + (len(v) - 5) // 2]]
            test_data = [item for k, v in data_dict.items() for item in v[5 + (len(v) - 5) // 2:]]

            # Check
            assert len(data) == len(train_data) + len(dev_data) + len(test_data)
            assert set([d[0] for d in data]) == set([d[0] for d in train_data + dev_data + test_data])

            # Save

            for set_type, set_data in zip([
                "train", "dev", "test"
            ], [
                train_data, dev_data, test_data
            ]):
                with open(f"data/ARSC-Yu/split{split_ix}/{label}.t{i}.{set_type}", "w") as file:
                    for line in set_data:
                        file.write("\t".join(line) + "\n")
