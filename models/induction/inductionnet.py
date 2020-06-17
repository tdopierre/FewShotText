import json
import argparse
from utils.data import get_jsonl_data
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict
from tensorboardX import SummaryWriter
import numpy as np
from models.encoders.bert_encoder import BERTEncoder
import torch
import torch.nn as nn
import warnings
import logging
from utils.few_shot import create_episode

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class InductionNet(nn.Module):
    def __init__(self, encoder, hidden_dim: int = 768, ntl_n_slices: int = 100, n_routing_iter: int = 3):
        super(InductionNet, self).__init__()

        self.encoder = encoder

        self.ntl_n_slices: int = ntl_n_slices
        self.n_routing_iter: int = n_routing_iter
        self.hidden_dim = hidden_dim
        self.relation_module = NTLRelationModule(input_dim=self.hidden_dim, n_slice=self.ntl_n_slices).to(device)
        self.induction_module = InductionModule(input_dim=hidden_dim, n_routing_iter=self.n_routing_iter).to(device)

    def loss(self, sample):
        """
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
        xs = sample["xs"]  # support
        xq = sample["xq"]  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        x = [item for xs_ in xs for item in xs_] + [item for xq_ in xq for item in xq_]
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_query = z[n_class * n_support:]
        z_support = z[:n_class * n_support].view(n_class, n_support, z_dim)

        class_representatives = self.induction_module.forward(z_s=z_support)
        relation_module_scores = self.relation_module.forward(z_q=z_query, z_c=class_representatives)
        true_labels = torch.zeros_like(relation_module_scores).to(device)

        for ix_class, class_query_sentences in enumerate(xq):
            for ix_sentence, sentence in enumerate(class_query_sentences):
                true_labels[ix_class * n_query + ix_sentence, ix_class] = 1

        loss_fn = nn.CrossEntropyLoss()

        loss_val = loss_fn(relation_module_scores, true_labels.argmax(1))
        acc_val = (true_labels.argmax(1) == relation_module_scores.argmax(1)).float().mean()

        return loss_val, {
            "loss": loss_val.item(),
            "acc": acc_val.item(),
            "y_hat": relation_module_scores.argmax(1).cpu().detach().numpy()
        }

    def train_step(self, optimizer, data_dict: Dict[str, List[str]], n_support, n_classes, n_query):

        episode = create_episode(
            data_dict=data_dict,
            n_support=n_support,
            n_classes=n_classes,
            n_query=n_query
        )

        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step(self, data_dict, n_support, n_classes, n_query, n_episodes=1000):
        accuracies = list()
        losses = list()
        self.eval()
        for i in range(n_episodes):
            episode = create_episode(
                data_dict=data_dict,
                n_support=n_support,
                n_classes=n_classes,
                n_query=n_query
            )

            with torch.no_grad():
                loss, loss_dict = self.loss(episode)

            accuracies.append(loss_dict["acc"])
            losses.append(loss_dict["loss"])

        return {
            "loss": np.mean(losses),
            "acc": np.mean(accuracies)
        }


class InductionModule(nn.Module):
    def __init__(self, input_dim: int, n_routing_iter: int = 3):
        super(InductionModule, self).__init__()
        self.input_dim: int = input_dim
        self.n_routing_iter: int = n_routing_iter

        # Init Ws, bs
        self.Ws_bs = nn.Linear(input_dim, input_dim)
        Ws = np.random.randn(input_dim, input_dim)
        Ws = Ws / np.linalg.norm(Ws)
        self.Ws = torch.Tensor(Ws).to(device)

    @staticmethod
    def squash(x):
        return (x / x.norm(dim=1)[:, None]) * ((x.norm(dim=1) ** 2) / (1 + (x.norm(dim=1) ** 2)))[:, None]

    def forward(self, z_s):
        """
        :param z_s: embedding of support samples, shape=(C, K, hidden_dim)
        :return:
        """
        C, K, _ = z_s.size()
        class_representatives: List[torch.Tensor] = list()

        for i in range(C):
            z = z_s[i]
            z_squashed = self.squash(z)
            b_i = torch.zeros(K).to(device)
            c_i = None
            for iteration in range(self.n_routing_iter):
                d_i = nn.Softmax(dim=-1)(b_i)
                c_i = d_i @ z_squashed
                c_i = self.squash(c_i.view(1, -1))
                b_i += (z_squashed @ c_i.view(-1, 1)).view(-1)

            class_representatives.append(c_i)
        class_representatives = torch.cat(class_representatives).to(device)
        return class_representatives


class NTLRelationModule(nn.Module):
    def __init__(self, input_dim, n_slice=100):
        super(NTLRelationModule, self).__init__()
        self.n_slice = n_slice
        M = np.random.randn(n_slice, input_dim, input_dim)
        M = M / np.linalg.norm(M, axis=(1, 2))[:, None, None]
        self.M = torch.Tensor(M).to(device)
        self.M.requires_grad = True
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(n_slice, 1)

    def forward(self, z_q, z_c):
        n_query = z_q.size(0)
        n_class = z_c.size(0)

        v = self.dropout(nn.ReLU()(torch.cat([(z_q @ m @ z_c.T).unsqueeze(-1) for m in self.M], dim=-1).view(-1, self.n_slice)))
        r_logit = self.fc(v).view(n_query, n_class)
        return r_logit


def run_induction(
        train_path: str,
        model_name_or_path: str,
        n_support: int,
        n_query: int,
        n_classes: int,
        valid_path: str = None,
        test_path: str = None,
        output_path: str = f"runs/{now()}",
        max_iter: int = 10000,
        evaluate_every: int = 100,
        early_stop: int = None,
        n_test_episodes: int = 1000,
        log_every: int = 10,
        relation_module_type: str = "base",
        ntl_n_slices: int = 100,
        n_routing_iter: int = 3
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

    if valid_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
    if test_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()

    def raw_data_to_labels_dict(data, shuffle=True):
        labels_dict = collections.defaultdict(list)
        for item in data:
            labels_dict[item["label"]].append(item["sentence"])
        labels_dict = dict(labels_dict)
        if shuffle:
            for key, val in labels_dict.items():
                random.shuffle(val)
        return labels_dict

    # Load model
    bert = BERTEncoder(model_name_or_path).to(device)
    matching_net = InductionNet(
        encoder=bert,
        relation_module_type=relation_module_type,
        ntl_n_slices=ntl_n_slices,
        n_routing_iter=n_routing_iter
    )
    optimizer = torch.optim.Adam(matching_net.parameters(), lr=2e-5)

    # Load data
    train_data = get_jsonl_data(train_path)
    train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
    logger.info(f"train labels: {train_data_dict.keys()}")

    if valid_path:
        valid_data = get_jsonl_data(valid_path)
        valid_data_dict = raw_data_to_labels_dict(valid_data, shuffle=True)
        logger.info(f"valid labels: {valid_data_dict.keys()}")
    else:
        valid_data_dict = None

    if test_path:
        test_data = get_jsonl_data(test_path)
        test_data_dict = raw_data_to_labels_dict(test_data, shuffle=True)
        logger.info(f"test labels: {test_data_dict.keys()}")
    else:
        test_data_dict = None

    train_accuracies = list()
    train_losses = list()
    n_eval_since_last_best = 0
    best_valid_acc = 0.0

    for step in range(max_iter):
        loss, loss_dict = matching_net.train_step(
            optimizer=optimizer,
            data_dict=train_data_dict,
            n_support=n_support,
            n_query=n_query,
            n_classes=n_classes
        )
        train_accuracies.append(loss_dict["acc"])
        train_losses.append(loss_dict["loss"])

        # Logging
        if (step + 1) % log_every == 0:
            train_writer.add_scalar(tag="loss", scalar_value=np.mean(train_losses), global_step=step)
            train_writer.add_scalar(tag="accuracy", scalar_value=np.mean(train_accuracies), global_step=step)
            logger.info(f"train | loss: {np.mean(train_losses):.4f} | acc: {np.mean(train_accuracies):.4f}")
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": "accuracy",
                        "value": np.mean(train_accuracies)
                    },
                    {
                        "tag": "loss",
                        "value": np.mean(train_losses)
                    }

                ],
                "global_step": step
            })

            train_accuracies = list()
            train_losses = list()

        if valid_path or test_path:
            if (step + 1) % evaluate_every == 0:
                for path, writer, set_type, set_data in zip(
                        [valid_path, test_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_data_dict, test_data_dict]
                ):
                    if path:
                        set_results = matching_net.test_step(
                            data_dict=set_data,
                            n_support=n_support,
                            n_query=n_query,
                            n_classes=n_classes,
                            n_episodes=n_test_episodes
                        )
                        writer.add_scalar(tag="loss", scalar_value=set_results["loss"], global_step=step)
                        writer.add_scalar(tag="accuracy", scalar_value=set_results["acc"], global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": "accuracy",
                                    "value": set_results["acc"]
                                },
                                {
                                    "tag": "loss",
                                    "value": set_results["loss"]
                                }

                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | loss: {set_results['loss']:.4f} | acc: {set_results['acc']:.4f}")
                        if set_type == "valid":
                            if set_results["acc"] > best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")

                if early_stop and n_eval_since_last_best >= early_stop:
                    print(f"Early-stopping.")
                    break
    with open(os.path.join(output_path, "metrics.json"), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--test-path", type=str, default=None, help="Path to testing data")

    parser.add_argument("--output-path", type=str, default=f"runs/{now()}")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, help="Number of support points for each class", required=True)
    parser.add_argument("--n-query", type=int, help="Number of query points for each class", required=True)
    parser.add_argument("--n-classes", type=int, help="Number of classes per episode", required=True)
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Relation Network-specific
    parser.add_argument("--relation-module-type", type=str, required=True, help="Which relation module to use")
    parser.add_argument("--ntl-n-slices", type=int, default=100, help="Number of matrices to use in NTL")
    parser.add_argument("--n-routing-iter", type=int, default=3, help="Number of routing iterations in the induction module")

    args = parser.parse_args()

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.train_path, args.valid_path, args.test_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_induction(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        output_path=args.output_path,

        model_name_or_path=args.model_name_or_path,

        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        n_test_episodes=args.n_test_episodes,

        max_iter=args.max_iter,
        evaluate_every=args.evaluate_every,

        relation_module_type=args.relation_module_type,
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == "__main__":
    main()
