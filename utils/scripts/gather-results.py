import numpy as np
import os
from utils.data import get_jsonl_data
import json


def get_results(root_path: str = "./runs"):
    out = list()
    for root, folders, files in os.walk(root_path):
        for file in files:

            # General case
            if file == "metrics.json":
                with open(os.path.join(root, file), "r") as f:
                    metrics = json.load(f)
                    test_acc = max([
                        m['value']
                        for d in metrics["test"]
                        for m in d['metrics']
                        if m['tag'] == "accuracy"
                    ])
                with open(os.path.join(root, "config.json"), "r") as f:
                    config = json.load(f)
                config.update({
                    "test_acc": test_acc,
                    "method": root.split('/')[-2]
                })

                out.append(config)

            # Baseline case
            if file == "test_metrics.json":
                with open(os.path.join(root, file), "r") as f:
                    metrics = json.load(f)
                    test_acc = np.mean([m['acc'] for m in metrics])
                with open(os.path.join(root, "config.json"), "r") as f:
                    config = json.load(f)

                config.update({
                    "test_acc": test_acc,
                    "method": root.split('/')[-2]
                })

                out.append(config)
    return out


if __name__ == "__main__":
    res = get_results()
    import pandas as pd

    pd.DataFrame(res).to_excel("res.xlsx", index=False)
