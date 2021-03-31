import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import json
import argparse

from transformers import AutoTokenizer

from models.encoders.bert_encoder import BERTEncoder
from paraphrase.modeling import UnigramRandomDropParaphraseBatchPreparer, DBSParaphraseModel, BigramDropParaphraseBatchPreparer, BaseParaphraseBatchPreparer
from paraphrase.utils.data import FewShotDataset, FewShotSSLParaphraseDataset, FewShotSSLFileDataset
from utils.data import get_jsonl_data, FewShotDataLoader
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict, Callable, Union
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.autograd import Variable
import warnings
import logging
from utils.few_shot import create_episode, create_ARSC_test_episode, create_ARSC_train_episode
from utils.math import euclidean_dist, cosine_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ProtAugmentNet(nn.Module):
    def __init__(self, encoder, metric="euclidean"):
        super(ProtAugmentNet, self).__init__()

        self.encoder: BERTEncoder = encoder
        self.metric = metric
        assert self.metric in ('euclidean', 'cosine')

    def loss(self, sample, supervised_loss_share: float = 0):
        """
        :param supervised_loss_share: share of supervised loss in total loss
        :param sample: {
            "xs": [
                [support_A_1, support_A_2, ...],
                [support_B_1, support_B_2, ...],
                [support_C_1, support_C_2, ...],
                ...
            ],
            "xq": [
                [query_A_1, query_A_2, ...],
                [query_B_1, query_B_2, ...],
                [query_C_1, query_C_2, ...],
                ...
            ]
        }
        :return:
        """
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        has_augment = "x_augment" in sample
        if has_augment:
            augmentations = sample["x_augment"]

            n_augmentations_samples = len(sample["x_augment"])
            n_augmentations_per_sample = [len(item['tgt_texts']) for item in augmentations]
            assert len(set(n_augmentations_per_sample)) == 1
            n_augmentations_per_sample = n_augmentations_per_sample[0]

            supports = [item["sentence"] for xs_ in xs for item in xs_]
            queries = [item["sentence"] for xq_ in xq for item in xq_]
            augmentations_supports = [[item2 for item2 in item["tgt_texts"]] for item in sample["x_augment"]]
            augmentation_queries = [item["src_text"] for item in sample["x_augment"]]

            # Encode
            x = supports + queries + [item2 for item1 in augmentations_supports for item2 in item1] + augmentation_queries
            z = self.encoder.embed_sentences(x)
            z_dim = z.size(-1)

            # Dispatch
            z_support = z[:len(supports)].view(n_class, n_support, z_dim).mean(dim=[1])
            z_query = z[len(supports):len(supports) + len(queries)]
            z_aug_support = (z[len(supports) + len(queries):len(supports) + len(queries) + n_augmentations_per_sample * n_augmentations_samples]
                             .view(n_augmentations_samples, n_augmentations_per_sample, z_dim).mean(dim=[1]))
            z_aug_query = z[-len(augmentation_queries):]
        else:
            # When not using augmentations
            supports = [item["sentence"] for xs_ in xs for item in xs_]
            queries = [item["sentence"] for xq_ in xq for item in xq_]

            # Encode
            x = supports + queries
            z = self.encoder.embed_sentences(x)
            z_dim = z.size(-1)

            # Dispatch
            z_support = z[:len(supports)].view(n_class, n_support, z_dim).mean(dim=[1])
            z_query = z[len(supports):len(supports) + len(queries)]

        if self.metric == "euclidean":
            supervised_dists = euclidean_dist(z_query, z_support)
            if has_augment:
                unsupervised_dists = euclidean_dist(z_aug_query, z_aug_support)
        elif self.metric == "cosine":
            supervised_dists = (-cosine_similarity(z_query, z_support) + 1) * 5
            if has_augment:
                unsupervised_dists = (-cosine_similarity(z_aug_query, z_aug_support) + 1) * 5
        else:
            raise NotImplementedError

        from torch.nn import CrossEntropyLoss
        supervised_loss = CrossEntropyLoss()(-supervised_dists, target_inds.reshape(-1))
        _, y_hat_supervised = (-supervised_dists).max(1)
        acc_val_supervised = torch.eq(y_hat_supervised, target_inds.reshape(-1)).float().mean()

        if has_augment:
            # Unsupervised loss
            unsupervised_target_inds = torch.range(0, n_augmentations_samples - 1).to(device).long()
            unsupervised_loss = CrossEntropyLoss()(-unsupervised_dists, unsupervised_target_inds)
            _, y_hat_unsupervised = (-unsupervised_dists).max(1)
            acc_val_unsupervised = torch.eq(y_hat_unsupervised, unsupervised_target_inds.reshape(-1)).float().mean()

            # Final loss
            assert 0 <= supervised_loss_share <= 1
            final_loss = (supervised_loss_share) * supervised_loss + (1 - supervised_loss_share) * unsupervised_loss

            return final_loss, {
                "metrics": {
                    "supervised_acc": acc_val_supervised.item(),
                    "unsupervised_acc": acc_val_unsupervised.item(),
                    "supervised_loss": supervised_loss.item(),
                    "unsupervised_loss": unsupervised_loss.item(),
                    "supervised_loss_share": supervised_loss_share,
                    "final_loss": final_loss.item(),
                },
                "supervised_dists": supervised_dists,
                "unsupervised_dists": unsupervised_dists,
                "target": target_inds
            }

        return supervised_loss, {
            "metrics": {
                "acc": acc_val_supervised.item(),
                "loss": supervised_loss.item(),
            },
            "dists": supervised_dists,
            "target": target_inds
        }

    def train_step(self, optimizer, episode, supervised_loss_share: float):
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode, supervised_loss_share=supervised_loss_share)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step(self, dataset: FewShotDataset, n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode = dataset.get_episode()

            with torch.no_grad():
                loss, loss_dict = self.loss(episode, supervised_loss_share=1)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }


def run_proto(
        # Compulsory!
        data_path: str,
        train_labels_path: str,
        model_name_or_path: str,

        # Few-shot Stuff
        n_support: int,
        n_query: int,
        n_classes: int,
        metric: str = "euclidean",

        # Optional path to augmented data
        unlabeled_path: str = None,

        # Path training data ONLY (optional)
        train_path: str = None,

        # Validation & test
        valid_labels_path: str = None,
        test_labels_path: str = None,
        evaluate_every: int = 100,
        n_test_episodes: int = 1000,

        # Logging & Saving
        output_path: str = f'runs/{now()}',
        log_every: int = 10,

        # Training stuff
        max_iter: int = 10000,
        early_stop: int = None,

        # Augmentation & paraphrase
        n_augmentation: int = 5,
        paraphrase_model_name_or_path: str = None,
        paraphrase_tokenizer_name_or_path: str = None,
        paraphrase_num_beams: int = None,
        paraphrase_beam_group_size: int = None,
        paraphrase_diversity_penalty: float = None,
        paraphrase_filtering_strategy: str = None,
        paraphrase_drop_strategy: str = None,
        paraphrase_drop_chance_speed: str = None,
        paraphrase_drop_chance_auc: float = None,
        supervised_loss_share_fn: Callable[[int, int], float] = lambda x, y: 1 - (x / y),

        augmentation_data_path: str = None
):
    if output_path:
        if os.path.exists(output_path) and len(os.listdir(output_path)):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    # ----------
    # Load model
    # ----------
    bert = BERTEncoder(model_name_or_path).to(device)
    protonet: ProtAugmentNet = ProtAugmentNet(encoder=bert, metric=metric)
    optimizer = torch.optim.Adam(protonet.parameters(), lr=2e-5)

    # ------------------
    # Load Train Dataset
    # ------------------
    if augmentation_data_path:
        # If an augmentation data path is provided, uses those pre-generated augmentations
        train_dataset = FewShotSSLFileDataset(
            data_path=train_path if train_path else data_path,
            labels_path=train_labels_path,
            n_classes=n_classes,
            n_support=n_support,
            n_query=n_query,
            n_unlabeled=n_augmentation,
            unlabeled_file_path=augmentation_data_path,
        )
    else:
        # ---------------------
        # Load paraphrase model
        # ---------------------
        paraphrase_model_device = torch.device("cpu") if "20newsgroup" in data_path else torch.device("cuda")
        logger.info(f"Paraphrase model device: {paraphrase_model_device}")
        paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_tokenizer_name_or_path)
        if paraphrase_drop_strategy == "unigram":
            paraphrase_batch_preparer = UnigramRandomDropParaphraseBatchPreparer(
                tokenizer=paraphrase_tokenizer,
                auc=paraphrase_drop_chance_auc,
                drop_chance_speed=paraphrase_drop_chance_speed,
                device=paraphrase_model_device
            )
        elif paraphrase_drop_strategy == "bigram":
            paraphrase_batch_preparer = BigramDropParaphraseBatchPreparer(tokenizer=paraphrase_tokenizer, device=paraphrase_model_device)
        else:
            paraphrase_batch_preparer = BaseParaphraseBatchPreparer(tokenizer=paraphrase_tokenizer, device=paraphrase_model_device)

        paraphrase_model = DBSParaphraseModel(
            model_name_or_path=paraphrase_model_name_or_path,
            tok_name_or_path=paraphrase_tokenizer_name_or_path,
            num_beams=paraphrase_num_beams,
            beam_group_size=paraphrase_beam_group_size,
            diversity_penalty=paraphrase_diversity_penalty,
            filtering_strategy=paraphrase_filtering_strategy,
            paraphrase_batch_preparer=paraphrase_batch_preparer,
            device=paraphrase_model_device
        )

        train_dataset = FewShotSSLParaphraseDataset(
            data_path=train_path if train_path else data_path,
            labels_path=train_labels_path,
            n_classes=n_classes,
            n_support=n_support,
            n_query=n_query,
            n_unlabeled=n_augmentation,
            unlabeled_file_path=unlabeled_path,
            paraphrase_model=paraphrase_model
        )
    logger.info(f"Train dataset has {len(train_dataset)} items")

    # ---------
    # Load data
    # ---------
    logger.info(f"train labels: {train_dataset.data.keys()}")
    valid_dataset: FewShotDataset = None
    if valid_labels_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
        valid_dataset = FewShotDataset(data_path=data_path, labels_path=valid_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"valid labels: {valid_dataset.data.keys()}")
        assert len(set(valid_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    test_dataset: FewShotDataset = None
    if test_labels_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()
        test_dataset = FewShotDataset(data_path=data_path, labels_path=test_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"test labels: {test_dataset.data.keys()}")
        assert len(set(test_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0

    for step in range(max_iter):
        episode = train_dataset.get_episode()

        supervised_loss_share = supervised_loss_share_fn(step, max_iter)
        loss, loss_dict = protonet.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share)

        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            for key, value in train_metrics.items():
                train_writer.add_scalar(tag=key, scalar_value=np.mean(value), global_step=step)
            logger.info(f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": key,
                        "value": np.mean(value)
                    }
                    for key, value in train_metrics.items()
                ],
                "global_step": step
            })

            train_metrics = collections.defaultdict(list)

        if valid_labels_path or test_labels_path:
            if (step + 1) % evaluate_every == 0:
                for labels_path, writer, set_type, set_dataset in zip(
                        [valid_labels_path, test_labels_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_dataset, test_dataset]
                ):
                    if set_dataset:

                        set_results = protonet.test_step(
                            dataset=set_dataset,
                            n_episodes=n_test_episodes
                        )

                        for key, val in set_results.items():
                            writer.add_scalar(tag=key, scalar_value=val, global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": key,
                                    "value": val
                                }
                                for key, val in set_results.items()
                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in set_results.items()]))
                        if set_type == "valid":
                            if set_results["acc"] > best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    break

    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to data")
    parser.add_argument("--train-labels-path", type=str, required=True, help="Path to train labels")
    parser.add_argument("--train-path", type=str, help="Path to training data (if provided, picks training data from this path instead of --data-path")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use", choices=("euclidean", "cosine"))

    # Path to augmented data
    parser.add_argument("--unlabeled-path", type=str, required=True, help="Path to data containing augmentations used for consistency")

    # Validation & test
    parser.add_argument("--valid-labels-path", type=str, required=True, help="Path to valid labels")
    parser.add_argument("--test-labels-path", type=str, required=True, help="Path to test labels")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Logging & Saving
    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")

    # Training stuff
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Augmentation & Paraphrase
    parser.add_argument("--n-augmentation", type=int, help="Number of unlabeled data points per class (proto++)", default=0)
    parser.add_argument("--paraphrase-model-name-or-path", type=str)
    parser.add_argument("--paraphrase-tokenizer-name-or-path", type=str)
    parser.add_argument("--paraphrase-num-beams", type=int)
    parser.add_argument("--paraphrase-beam-group-size", type=int)
    parser.add_argument("--paraphrase-diversity-penalty", type=float)
    parser.add_argument("--paraphrase-filtering-strategy", type=str)
    parser.add_argument("--paraphrase-drop-strategy", type=str)
    parser.add_argument("--paraphrase-drop-chance-speed", type=str)
    parser.add_argument("--paraphrase-drop-chance-auc", type=float)

    # Augmentation file path (optional, but if provided it will be used)
    parser.add_argument("--augmentation-data-path", type=str)

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")

    args = parser.parse_args()
    logger.debug(f"Received args: {json.dumps(args.__dict__, sort_keys=True, ensure_ascii=False, indent=1)}")
    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.data_path, args.train_labels_path, args.valid_labels_path, args.test_labels_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Create supervised_loss_share_fn
    def get_supervised_loss_share_fn(supervised_loss_share_power: Union[int, float]) -> Callable[[int, int], float]:
        def _supervised_loss_share_fn(current_step: int, max_steps: int) -> float:
            assert current_step <= max_steps
            return 1 - (current_step / max_steps) ** supervised_loss_share_power

        return _supervised_loss_share_fn

    supervised_loss_share_fn = get_supervised_loss_share_fn(args.supervised_loss_share_power)

    # Run
    run_proto(
        data_path=args.data_path,
        train_labels_path=args.train_labels_path,
        train_path=args.train_path,
        model_name_or_path=args.model_name_or_path,
        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        metric=args.metric,
        unlabeled_path=args.unlabeled_path,

        valid_labels_path=args.valid_labels_path,
        test_labels_path=args.test_labels_path,
        evaluate_every=args.evaluate_every,
        n_test_episodes=args.n_test_episodes,

        output_path=args.output_path,
        log_every=args.log_every,
        max_iter=args.max_iter,
        early_stop=args.early_stop,

        n_augmentation=args.n_augmentation,
        paraphrase_model_name_or_path=args.paraphrase_model_name_or_path,
        paraphrase_tokenizer_name_or_path=args.paraphrase_tokenizer_name_or_path,
        paraphrase_num_beams=args.paraphrase_num_beams,
        paraphrase_beam_group_size=args.paraphrase_beam_group_size,
        paraphrase_filtering_strategy=args.paraphrase_filtering_strategy,
        paraphrase_drop_strategy=args.paraphrase_drop_strategy,
        paraphrase_drop_chance_speed=args.paraphrase_drop_chance_speed,
        paraphrase_drop_chance_auc=args.paraphrase_drop_chance_auc,
        supervised_loss_share_fn=supervised_loss_share_fn,

        augmentation_data_path=args.augmentation_data_path
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
