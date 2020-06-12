from typing import List

import torch.nn as nn
import logging
import warnings
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BERTEncoder(nn.Module):
    def __init__(self, config_name_or_path):
        super(BERTEncoder, self).__init__()
        logger.info(f"Loading Encoder @ {config_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.bert = AutoModel.from_pretrained(config_name_or_path)
        logger.info(f"Encoder loaded.")

    def forward(self, sentences: List[str]):
        batch_size = 2
        if len(sentences) > batch_size:
            return torch.cat([self.forward(sentences[i:i + batch_size]) for i in range(0, len(sentences), batch_size)], 0)
        encoded_plus = [self.tokenizer.encode_plus(s, max_length=256) for s in sentences]
        max_len = max([len(e['input_ids']) for e in encoded_plus])

        input_ids = list()
        attention_masks = list()
        token_type_ids = list()

        for e in encoded_plus:
            e['input_ids'] = e['input_ids'][:max_len]
            e['token_type_ids'] = e['token_type_ids'][:max_len]
            pad_len = max_len - len(e['input_ids'])
            input_ids.append(e['input_ids'] + pad_len * [self.tokenizer.pad_token_id])
            attention_masks.append([1 for _ in e['input_ids']] + [0] * pad_len)
            token_type_ids.append(e['token_type_ids'] + [0] * pad_len)

        _, x = self.bert.forward(input_ids=torch.Tensor(input_ids).long().to(device),
                                 attention_mask=torch.Tensor(attention_masks).long().to(device),
                                 token_type_ids=torch.Tensor(token_type_ids).long().to(device))
        return x
