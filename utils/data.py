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
