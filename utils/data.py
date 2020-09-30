import numpy as np
import random
import collections
import os
import json
from typing import List, Dict


def get_jsonl_data(jsonl_path: str):
    out = list()
    with open(jsonl_path, 'r') as file:
        for line in file:
            j = json.loads(line.strip())
            out.append(j)
    return out


def write_jsonl_data(jsonl_data: List[Dict], jsonl_path: str, force=False):
    if os.path.exists(jsonl_path) and not force:
        raise FileExistsError
    with open(jsonl_path, 'w') as file:
        for line in jsonl_data:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')


def get_tsv_data(tsv_path: str, label: str = None):
    out = list()
    with open(tsv_path, "r") as file:
        for line in file:
            line = line.strip().split('\t')
            if not label:
                label = tsv_path.split('/')[-1]

            out.append({
                "sentence": line[0],
                "label": label + str(line[1])
            })
    return out


class FewShotDataLoader:
    def __init__(self, file_path):
        self.raw_data = get_jsonl_data(file_path)
        self.data_dict = self.raw_data_to_dict(self.raw_data, shuffle=True)

    @staticmethod
    def raw_data_to_dict(data, shuffle=True):
        labels_dict = collections.defaultdict(list)
        for item in data:
            labels_dict[item['label']].append(item)
        labels_dict = dict(labels_dict)
        if shuffle:
            for key, val in labels_dict.items():
                random.shuffle(val)
        return labels_dict

    def create_episode(self, n_support: int = 0, n_classes: int = 0, n_query: int = 0, n_unlabeled: int = 0, n_augment: int = 0):
        episode = dict()
        if n_classes:
            n_classes = min(n_classes, len(self.data_dict.keys()))
            rand_keys = np.random.choice(list(self.data_dict.keys()), n_classes, replace=False)

            assert min([len(val) for val in self.data_dict.values()]) >= n_support + n_query + n_unlabeled

            for key, val in self.data_dict.items():
                random.shuffle(val)

            if n_support:
                episode["xs"] = [[self.data_dict[k][i] for i in range(n_support)] for k in rand_keys]
            if n_query:
                episode["xq"] = [[self.data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys]

            if n_unlabeled:
                episode['xu'] = [item for k in rand_keys for item in self.data_dict[k][n_support + n_query:n_support + n_query + n_unlabeled]]

        if n_augment:
            augmentations = list()
            already_done = list()
            for i in range(n_augment):
                # Draw a random label
                key = random.choice(list(self.data_dict.keys()))
                # Draw a random data index
                ix = random.choice(range(len(self.data_dict[key])))
                # If already used, re-sample
                while (key, ix) in already_done:
                    key = random.choice(list(self.data_dict.keys()))
                    ix = random.choice(range(len(self.data_dict[key])))
                already_done.append((key, ix))
                if "augmentations" not in self.data_dict[key][ix]:
                    raise KeyError(f"Input data {self.data_dict[key][ix]} does not contain any augmentations / is not properly formatted.")
                augmentations.append((self.data_dict[key][ix]))

            episode["x_augment"] = augmentations

        return episode
