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
from utils.few_shot import create_episode, create_ARSC_train_episode, create_ARSC_test_episode

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')
torch.autograd.set_detect_anomaly(True)

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

        # MSE LOSS
        # relation_module_scores = torch.sigmoid(relation_module_scores)
        # loss_fn = nn.MSELoss()
        # loss_val = loss_fn(relation_module_scores, true_labels)
        # acc_full = ((relation_module_scores > 0.5).float() == true_labels.float()).float().mean()
        # acc_exact = (((relation_module_scores > 0.5).float() - true_labels.float()).abs().max(dim=1)[0] == 0).float().mean()
        # acc_max = (relation_module_scores.argmax(1) == true_labels.argmax(1)).float().mean()
        #
        # return loss_val, {
        #     "loss": loss_val.item(),
        #     "metrics": {
        #         "loss": loss_val.item(),
        #         "acc_full": acc_full.item(),
        #         "acc_exact": acc_exact.item(),
        #         "acc_max": acc_max.item(),
        #         "acc": acc_max.item()
        #     },
        #     "y_hat": relation_module_scores.argmax(1).cpu().detach().numpy()
        # }

        # CE LOSS
        loss_fn = nn.CrossEntropyLoss()
        loss_val = loss_fn(relation_module_scores, true_labels.argmax(1))
        acc_val = (true_labels.argmax(1) == relation_module_scores.argmax(1)).float().mean()
        return loss_val, {
            "loss": loss_val.item(),
            "metrics": {
                "loss": loss_val.item(),
                "acc": acc_val.item()
            },
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
        metrics = collections.defaultdict(list)
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

            for key, value in loss_dict["metrics"].items():
                metrics[key].append(value)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }

    def train_step_ARSC(self, data_path: str, optimizer):
        episode = create_ARSC_train_episode(prefix=data_path, n_support=5, n_query=5)

        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step_ARSC(self, data_path: str, n_episodes=1000, set_type="test"):
        assert set_type in ("dev", "test")
        metrics = collections.defaultdict(list)
        self.eval()
        for i in range(n_episodes):
            episode = create_ARSC_test_episode(prefix=data_path, n_query=5, set_type=set_type)

            with torch.no_grad():
                loss, loss_dict = self.loss(episode)

            for key, value in loss_dict["metrics"].items():
                metrics[key].append(value)

        return {
            key: np.mean(value) for key, value in metrics.items()
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
        C, K, hidden_dim = z_s.size()
        class_representatives: List[torch.Tensor] = list()

        for i in range(C):
            z_squashed = self.squash(z_s[i])
            b_i = torch.autograd.Variable(torch.zeros(K)).to(device)
            for iteration in range(self.n_routing_iter):
                d_i = b_i.clone().softmax(dim=-1)
                c_i = torch.matmul(d_i, z_squashed)
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
        ntl_n_slices: int = 100,
        n_routing_iter: int = 3,
        arsc_format: bool = False,
        data_path: str = None
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
    induction_net = InductionNet(
        encoder=bert,
        ntl_n_slices=ntl_n_slices,
        n_routing_iter=n_routing_iter
    )
    optimizer = torch.optim.Adam(induction_net.parameters(), lr=2e-5)

    # Load data
    if not arsc_format:
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
    else:
        train_data_dict = None
        test_data_dict = None
        valid_data_dict = None

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0

    for step in range(max_iter):
        if not arsc_format:
            loss, loss_dict = induction_net.train_step(
                optimizer=optimizer,
                data_dict=train_data_dict,
                n_support=n_support,
                n_query=n_query,
                n_classes=n_classes
            )
        else:
            loss, loss_dict = induction_net.train_step_ARSC(
                optimizer=optimizer,
                data_path=data_path
            )

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

        if valid_path or test_path:
            if (step + 1) % evaluate_every == 0:
                for path, writer, set_type, set_data in zip(
                        [valid_path, test_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_data_dict, test_data_dict]
                ):
                    if path:
                        if not arsc_format:
                            set_results = induction_net.test_step(
                                data_dict=set_data,
                                n_support=n_support,
                                n_query=n_query,
                                n_classes=n_classes,
                                n_episodes=n_test_episodes
                            )
                        else:
                            set_results = induction_net.test_step_ARSC(
                                data_path=data_path,
                                n_episodes=n_test_episodes,
                                set_type={"valid": "dev", "test": "test"}[set_type]
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
                    print(f"Early-stopping.")
                    break
    with open(os.path.join(output_path, "metrics.json"), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--test-path", type=str, default=None, help="Path to testing data")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data (ARSC only)")

    parser.add_argument("--output-path", type=str, default=f"runs/{now()}")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Relation Network-specific
    parser.add_argument("--ntl-n-slices", type=int, default=100, help="Number of matrices to use in NTL")
    parser.add_argument("--n-routing-iter", type=int, default=3, help="Number of routing iterations in the induction module")

    # ARSC data
    parser.add_argument("--arsc-format", default=False, action="store_true", help="Using ARSC few-shot format")
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
        n_routing_iter=args.n_routing_iter,
        ntl_n_slices=args.ntl_n_slices,
        early_stop=args.early_stop,
        arsc_format=args.arsc_format,
        data_path=args.data_path
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == "__main__":
    main()
