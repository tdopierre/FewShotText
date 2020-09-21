import argparse
import json
import numpy as np


def get_arsc_results_from_baseline_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    test_accs = [
        np.mean([max([evaluation['test']['acc'] for evaluation in task]) for task in d['test']])
        for d in data if "test" in d
    ]
    valid_accs = [
        np.mean([max([evaluation['valid']['acc'] for evaluation in task]) for task in d['test']])
        for d in data if "test" in d
    ]

    return max(test_accs), max(valid_accs)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--path", type=str, required=True, help="path to baseline ARSC results (**/baseline_metrics.json)")
    args = args_parser.parse_args()

    test, valid = get_arsc_results_from_baseline_file(path=args.path)

    print(test)


if __name__ == "__main__":
    main()
