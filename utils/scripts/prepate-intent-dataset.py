import random
from utils.data import get_jsonl_data, write_jsonl_data, write_txt_data


def process_dataset(name):
    full = get_jsonl_data(f"data/{name}/full.jsonl")
    random.shuffle(full)

    labels = sorted(set([d['label'] for d in full]))
    n_labels = len(labels)
    random.seed(42)
    random.shuffle(labels)

    train_labels, valid_labels, test_labels = (
        labels[:int(n_labels / 3)],
        labels[int(n_labels / 3):int(2 * n_labels / 3)],
        labels[int(2 * n_labels / 3):]
    )

    write_jsonl_data([
        d for d in full if d['label'] in train_labels
    ], f"data/{name}/train.jsonl", force=True)
    write_txt_data(train_labels, f"data/{name}/labels.train.txt")

    write_jsonl_data([
        d for d in full if d['label'] in valid_labels
    ], f"data/{name}/valid.jsonl", force=True)
    write_txt_data(valid_labels, f"data/{name}/labels.valid.txt")

    write_jsonl_data([
        d for d in full if d['label'] in test_labels
    ], f"data/{name}/test.jsonl", force=True)
    write_txt_data(test_labels, f"data/{name}/labels.test.txt")


if __name__ == "__main__":
    for name in ('OOS', 'TREC28', 'Liu'):
        process_dataset(name)
