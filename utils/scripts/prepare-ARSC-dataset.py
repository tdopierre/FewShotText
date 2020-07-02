from utils.data import write_jsonl_data, get_jsonl_data
import collections
import json
import random
import tempfile
import requests
import os
import gzip
import shutil
import logging

import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

train_urls = [
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Tools_and_Home_Improvement_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Patio_Lawn_and_Garden_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz",
]
test_urls = [
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz",

]


def download_data(url, path: str):
    if not path:
        path = tempfile.mktemp()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Download
    logger.info(f"Loading data @ {url}")
    r = requests.get(url)

    logger.info(f"Saving data @ {path}")
    with open(path, 'wb') as f:
        f.write(r.content)


def url_to_label(url):
    return os.path.split(url)[-1][8:-10].lower()


def download_all():
    for url in train_urls + test_urls:
        label = url_to_label(url)
        path = f"data/ARSC/raw/{label}.json.gz"
        json_path = f"data/ARSC/raw/{label}.json"
        if not os.path.exists(json_path):
            download_data(url=url, path=path)


def path_to_sample(path, label, max_samples=5000):
    out = {
        f"{label}_0": [],
        f"{label}_1": [],
        f"{label}_2": [],
    }

    with open(path, "r") as file:
        for line in file:
            d = json.loads(line.strip())

            if d['overall'] == 5:
                d['label'] = f"{label}_2"
            elif d['overall'] == 4:
                d['label'] = f"{label}_1"
            elif d['overall'] in (2, 3):
                d['label'] = f"{label}_0"
            else:
                continue
            if len(out[d['label']]) < max_samples:
                out[d['label']].append({
                    "sentence": d['reviewText'],
                    "label": d['label']
                })

            if min([len(v) for v in out.values()]) == max_samples:
                break
    out_data = [item for v in out.values() for item in v]
    random.shuffle(out_data)
    return out_data


def data_to_sample(data, label, max_samples=5000):
    for d in data:
        if d['overall'] == 5:
            d['label'] = f"{label}_2"
        elif d['overall'] == 4:
            d['label'] = f"{label}_1"
        elif d['overall'] in (2, 3):
            d['label'] = f"{label}_0"
        else:
            pass
    data = [d for d in data if 'label' in d]
    random.shuffle(data)

    data_dict = collections.defaultdict(list)
    for d in data:
        data_dict[d['label']].append()

    sample = [item for k, v in data_dict.items() for item in v[:max_samples]]
    random.shuffle(sample)
    return sample


def create_dataset():
    # Train
    if not os.path.exists("data/ARSC/train.jsonl"):
        train_data = list()
        for url in tqdm.tqdm(train_urls):
            label = url_to_label(url)
            label_data = path_to_sample(f'data/ARSC/raw/{label}.json', label=label, max_samples=5000)
            train_data += label_data
        random.shuffle(train_data)
        os.makedirs('data/ARSC', exist_ok=True)
        write_jsonl_data(train_data, f'data/ARSC/train.jsonl')

    # Test
    if not os.path.exists("data/ARSC/test.jsonl"):
        test_data = list()
        for url in tqdm.tqdm(test_urls):
            label = url_to_label(url)
            label_data = path_to_sample(f'data/ARSC/raw/{label}.json', label=label, max_samples=5000)
            test_data += label_data
        random.shuffle(test_data)
        os.makedirs('data/ARSC', exist_ok=True)
        write_jsonl_data(test_data, f'data/ARSC/test.jsonl', force=True)


if __name__ == "__main__":
    download_all()
    create_dataset()
