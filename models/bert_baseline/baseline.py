import json
import argparse

import tqdm

from models.encoders.bert_encoder import BERTEncoder
from utils.data import get_jsonl_data
from utils.python import now, set_seeds
from utils.few_shot import create_ARSC_train_episode, get_ARSC_test_tasks
import random
import collections
import os
from typing import List, Dict
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import warnings
import logging
from utils.math import euclidean_dist, cosine_similarity

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BaselineNet(nn.Module):
    def __init__(
            self,
            encoder,
            is_pp: bool = False,
            hidden_dim: int = 768,
            metric: str = "cosine"
    ):
        super(BaselineNet, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=0.25).to(device)
        self.is_pp = is_pp
        self.hidden_dim = hidden_dim
        self.metric = metric
        assert self.metric in ("euclidean", "cosine")

    def train_ARSC_one_episode(
            self,
            data_path: str,
            n_iter: int = 100,
    ):
        self.train()
        episode = create_ARSC_train_episode(prefix=data_path, n_support=5, n_query=0, n_unlabeled=0)
        n_episode_classes = len(episode["xs"])
        loss_fn = nn.CrossEntropyLoss()
        episode_matrix = None
        episode_classifier = None
        if self.is_pp:
            with torch.no_grad():
                init_matrix = np.array([
                    [
                        self.encoder.forward([sentence]).squeeze().cpu().detach().numpy()
                        for sentence in episode["xs"][c]
                    ]
                    for c in range(n_episode_classes)
                ]).mean(1)

            episode_matrix = torch.Tensor(init_matrix).to(device)
            episode_matrix.requires_grad = True
            optimizer = torch.optim.Adam(list(self.parameters()) + [episode_matrix], lr=2e-5)
        else:
            episode_classifier = nn.Linear(in_features=self.hidden_dim, out_features=n_episode_classes).to(device)
            optimizer = torch.optim.Adam(list(self.parameters()) + list(episode_classifier.parameters()), lr=2e-5)

        # Train on support
        iter_bar = tqdm.tqdm(range(n_iter))
        losses = list()
        accuracies = list()

        for _ in iter_bar:
            optimizer.zero_grad()

            sentences = [sentence for sentence_list in episode["xs"] for sentence in sentence_list]
            labels = torch.Tensor([ix for ix, sl in enumerate(episode["xs"]) for _ in sl]).long().to(device)
            z = self.encoder(sentences)

            # z = batch_embeddings

            if self.is_pp:
                if self.metric == "cosine":
                    z = cosine_similarity(z, episode_matrix) * 5
                elif self.metric == "euclidean":
                    z = -euclidean_dist(z, episode_matrix)
                else:
                    raise NotImplementedError
            else:
                z = self.dropout(z)
                z = episode_classifier(z)

            loss = loss_fn(input=z, target=labels)
            acc = (z.argmax(1) == labels).float().mean()
            loss.backward()
            optimizer.step()
            iter_bar.set_description(f"{loss.item():.3f} | {acc.item():.3f}")
            losses.append(loss.item())
            accuracies.append(acc.item())
        return {
            "loss": np.mean(losses),
            "acc": np.mean(accuracies)
        }

    def run_ARSC(
            self,
            data_path: str,
            train_summary_writer: SummaryWriter = None,
            valid_summary_writer: SummaryWriter = None,
            test_summary_writer: SummaryWriter = None,
            n_episodes: int = 1000,
            n_train_iter: int = 100,
            train_eval_every: int = 100,
            n_test_iter: int = 1000,
            test_eval_every: int = 100,
    ):
        metrics = list()
        for episode_ix in range(n_episodes):
            output = self.train_ARSC_one_episode(data_path=data_path, n_iter=n_train_iter)
            episode_metrics = {
                "train": output
            }

            if train_summary_writer:
                train_summary_writer.add_scalar(tag=f'loss', global_step=episode_ix, scalar_value=output["loss"])
                train_summary_writer.add_scalar(tag=f'acc', global_step=episode_ix, scalar_value=output["acc"])

            # Running evaluation
            if (train_eval_every and (episode_ix + 1) % train_eval_every == 0) or (not train_eval_every and episode_ix + 1 == n_episodes):
                test_metrics = self.test_model_ARSC(
                    data_path=data_path,
                    valid_summary_writer=valid_summary_writer,
                    test_summary_writer=test_summary_writer,
                    n_iter=n_test_iter,
                    eval_every=test_eval_every
                )
                episode_metrics["test"] = test_metrics

            metrics.append(episode_metrics)
        return metrics

    def test_model_ARSC(
            self,
            data_path: str,
            n_iter: int = 1000,
            valid_summary_writer: SummaryWriter = None,
            test_summary_writer: SummaryWriter = None,
            eval_every: int = 100
    ):
        self.eval()

        tasks = get_ARSC_test_tasks(prefix=data_path)
        metrics = list()
        logger.info("Embedding sentences...")
        sentences_to_embed = [
            s
            for task in tasks
            for sentences_lists in task['xs'] + task['x_test'] + task['x_valid']
            for s in sentences_lists
        ]

        # sentence_to_embedding_dict = {s: np.random.randn(768) for s in tqdm.tqdm(sentences_to_embed)}
        sentence_to_embedding_dict = {s: self.encoder.forward([s]).cpu().detach().numpy().squeeze() for s in tqdm.tqdm(sentences_to_embed)}
        for ix_task, task in enumerate(tasks):
            task_metrics = list()

            n_episode_classes = 2
            loss_fn = nn.CrossEntropyLoss()
            episode_matrix = None
            episode_classifier = None
            if self.is_pp:
                with torch.no_grad():
                    init_matrix = np.array([
                        [
                            sentence_to_embedding_dict[sentence]
                            for sentence in task["xs"][c]
                        ]
                        for c in range(n_episode_classes)
                    ]).mean(1)

                episode_matrix = torch.Tensor(init_matrix).to(device)
                episode_matrix.requires_grad = True
                optimizer = torch.optim.Adam([episode_matrix], lr=2e-5)
            else:
                episode_classifier = nn.Linear(in_features=self.hidden_dim, out_features=n_episode_classes).to(device)
                optimizer = torch.optim.Adam(list(episode_classifier.parameters()), lr=2e-5)

            # Train on support
            iter_bar = tqdm.tqdm(range(n_iter))
            losses = list()
            accuracies = list()

            for iteration in iter_bar:
                optimizer.zero_grad()

                sentences = [sentence for sentence_list in task["xs"] for sentence in sentence_list]
                labels = torch.Tensor([ix for ix, sl in enumerate(task["xs"]) for _ in sl]).long().to(device)
                batch_embeddings = torch.Tensor([sentence_to_embedding_dict[s] for s in sentences]).to(device)
                # z = self.encoder(sentences)
                z = batch_embeddings

                if self.is_pp:
                    if self.metric == "cosine":
                        z = cosine_similarity(z, episode_matrix) * 5
                    elif self.metric == "euclidean":
                        z = -euclidean_dist(z, episode_matrix)
                    else:
                        raise NotImplementedError
                else:
                    z = self.dropout(z)
                    z = episode_classifier(z)

                loss = loss_fn(input=z, target=labels)
                acc = (z.argmax(1) == labels).float().mean()
                loss.backward()
                optimizer.step()
                iter_bar.set_description(f"{loss.item():.3f} | {acc.item():.3f}")
                losses.append(loss.item())
                accuracies.append(acc.item())

                if (eval_every and (iteration + 1) % eval_every == 0) or (not eval_every and iteration + 1 == n_iter):
                    self.eval()
                    if not self.is_pp:
                        episode_classifier.eval()

                    # --------------
                    #   VALIDATION
                    # --------------
                    valid_query_data_list = [
                        {"sentence": sentence, "label": label}
                        for label, sentences in enumerate(task["x_valid"])
                        for sentence in sentences
                    ]

                    valid_query_labels = torch.Tensor([d['label'] for d in valid_query_data_list]).long().to(device)
                    logits = list()
                    with torch.no_grad():
                        for ix in range(0, len(valid_query_data_list), 16):
                            batch = valid_query_data_list[ix:ix + 16]
                            batch_sentences = [d['sentence'] for d in batch]
                            batch_embeddings = torch.Tensor([sentence_to_embedding_dict[s] for s in batch_sentences]).to(device)
                            # z = self.encoder(batch_sentences)
                            z = batch_embeddings

                            if self.is_pp:
                                if self.metric == "cosine":
                                    z = cosine_similarity(z, episode_matrix) * 5
                                elif self.metric == "euclidean":
                                    z = -euclidean_dist(z, episode_matrix)
                                else:
                                    raise NotImplementedError
                            else:
                                z = episode_classifier(z)

                            logits.append(z)
                    logits = torch.cat(logits, dim=0)
                    y_hat = logits.argmax(1)

                    valid_loss = loss_fn(input=logits, target=valid_query_labels)
                    valid_acc = (y_hat == valid_query_labels).float().mean()

                    # --------------
                    #      TEST
                    # --------------
                    test_query_data_list = [
                        {"sentence": sentence, "label": label}
                        for label, sentences in enumerate(task["x_test"])
                        for sentence in sentences
                    ]

                    test_query_labels = torch.Tensor([d['label'] for d in test_query_data_list]).long().to(device)
                    logits = list()
                    with torch.no_grad():
                        for ix in range(0, len(test_query_data_list), 16):
                            batch = test_query_data_list[ix:ix + 16]
                            batch_sentences = [d['sentence'] for d in batch]
                            batch_embeddings = torch.Tensor([sentence_to_embedding_dict[s] for s in batch_sentences]).to(device)
                            # z = self.encoder(batch_sentences)
                            z = batch_embeddings

                            if self.is_pp:
                                if self.metric == "cosine":
                                    z = cosine_similarity(z, episode_matrix) * 5
                                elif self.metric == "euclidean":
                                    z = -euclidean_dist(z, episode_matrix)
                                else:
                                    raise NotImplementedError
                            else:
                                z = episode_classifier(z)

                            logits.append(z)
                    logits = torch.cat(logits, dim=0)
                    y_hat = logits.argmax(1)

                    test_loss = loss_fn(input=logits, target=test_query_labels)
                    test_acc = (y_hat == test_query_labels).float().mean()

                    # --RETURN METRICS
                    task_metrics.append({
                        "test": {
                            "loss": test_loss.item(),
                            "acc": test_acc.item()
                        },
                        "valid": {
                            "loss": valid_loss.item(),
                            "acc": valid_acc.item()
                        },
                        "step": iteration + 1
                    })
                    # if valid_summary_writer:
                    #     valid_summary_writer.add_scalar(tag=f'loss', global_step=ix_task, scalar_value=valid_loss.item())
                    #     valid_summary_writer.add_scalar(tag=f'acc', global_step=ix_task, scalar_value=valid_acc.item())
                    # if test_summary_writer:
                    #     test_summary_writer.add_scalar(tag=f'loss', global_step=ix_task, scalar_value=test_loss.item())
                    #     test_summary_writer.add_scalar(tag=f'acc', global_step=ix_task, scalar_value=test_acc.item())
            metrics.append(task_metrics)
        return metrics

    def train_model(
            self,
            data_dict: Dict[str, List[str]],
            summary_writer: SummaryWriter = None,
            n_epoch: int = 400,
            batch_size: int = 16,
            log_every: int = 10):
        self.train()

        training_classes = sorted(set(data_dict.keys()))
        n_training_classes = len(training_classes)
        class_to_ix = {c: ix for ix, c in enumerate(training_classes)}
        training_data_list = [{"sentence": sentence, "label": label} for label, sentences in data_dict.items() for sentence in sentences]

        training_matrix = None
        training_classifier = None

        if self.is_pp:
            training_matrix = torch.randn(n_training_classes, self.hidden_dim, requires_grad=True, device=device)
            optimizer = torch.optim.Adam(list(self.parameters()) + [training_matrix], lr=2e-5)
        else:
            training_classifier = nn.Linear(in_features=self.hidden_dim, out_features=n_training_classes).to(device)
            optimizer = torch.optim.Adam(list(self.parameters()) + list(training_classifier.parameters()), lr=2e-5)

        n_samples = len(training_data_list)
        loss_fn = nn.CrossEntropyLoss()
        global_step = 0

        # Metrics
        training_losses = list()
        training_accuracies = list()

        for _ in tqdm.tqdm(range(n_epoch)):
            random.shuffle(training_data_list)
            for ix in tqdm.tqdm(range(0, n_samples, batch_size)):
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                batch_items = training_data_list[ix:ix + batch_size]
                batch_sentences = [d['sentence'] for d in batch_items]
                batch_labels = torch.Tensor([class_to_ix[d['label']] for d in batch_items]).long().to(device)
                z = self.encoder(batch_sentences)
                if self.is_pp:
                    if self.metric == "cosine":
                        z = cosine_similarity(z, training_matrix) * 5
                    elif self.metric == "euclidean":
                        z = -euclidean_dist(z, training_matrix)
                    else:
                        raise NotImplementedError
                else:
                    z = self.dropout(z)
                    z = training_classifier(z)
                loss = loss_fn(input=z, target=batch_labels)
                acc = (z.argmax(1) == batch_labels).float().mean()
                loss.backward()
                optimizer.step()

                global_step += 1
                training_losses.append(loss.item())
                training_accuracies.append(acc.item())
                if (global_step % log_every) == 0:
                    if summary_writer:
                        summary_writer.add_scalar(tag="loss", global_step=global_step, scalar_value=np.mean(training_losses))
                        summary_writer.add_scalar(tag="acc", global_step=global_step, scalar_value=np.mean(training_accuracies))
                    # Empty metrics
                    training_losses = list()
                    training_accuracies = list()

    def test_one_episode(
            self,
            support_data_dict: Dict[str, List[str]],
            query_data_dict: Dict[str, List[str]],
            sentence_to_embedding_dict: Dict,
            batch_size: int = 4,
            n_iter: int = 1000,
            summary_writer: SummaryWriter = None,
            summary_tag_prefix: str = None,
    ):

        # Check data integrity
        assert set(support_data_dict.keys()) == set(query_data_dict.keys())

        # Freeze encoder
        self.encoder.eval()

        episode_classes = sorted(set(support_data_dict.keys()))
        n_episode_classes = len(episode_classes)
        class_to_ix = {c: ix for ix, c in enumerate(episode_classes)}
        ix_to_class = {ix: c for ix, c in enumerate(episode_classes)}
        support_data_list = [{"sentence": sentence, "label": label} for label, sentences in support_data_dict.items() for sentence in sentences]
        support_data_list = (support_data_list * batch_size * n_iter)[:(batch_size * n_iter)]

        loss_fn = nn.CrossEntropyLoss()
        episode_matrix = None
        episode_classifier = None
        if self.is_pp:
            init_matrix = np.array([
                [
                    sentence_to_embedding_dict[sentence].ravel()
                    for sentence in support_data_dict[ix_to_class[c]]
                ]
                for c in range(n_episode_classes)
            ]).mean(1)

            episode_matrix = torch.Tensor(init_matrix).to(device)
            episode_matrix.requires_grad = True
            optimizer = torch.optim.Adam([episode_matrix], lr=1e-3)
        else:
            episode_classifier = nn.Linear(in_features=self.hidden_dim, out_features=n_episode_classes).to(device)
            optimizer = torch.optim.Adam(list(episode_classifier.parameters()), lr=1e-3)

        # Train on support
        iter_bar = tqdm.tqdm(range(n_iter))
        for iteration in iter_bar:
            optimizer.zero_grad()

            batch = support_data_list[iteration * batch_size: iteration * batch_size + batch_size]
            batch_sentences = [d['sentence'] for d in batch]
            batch_embeddings = torch.Tensor([sentence_to_embedding_dict[s] for s in batch_sentences]).to(device)
            batch_labels = torch.Tensor([class_to_ix[d['label']] for d in batch]).long().to(device)
            # z = self.encoder(batch_sentences)
            z = batch_embeddings

            if self.is_pp:
                if self.metric == "cosine":
                    z = cosine_similarity(z, episode_matrix) * 5
                elif self.metric == "euclidean":
                    z = -euclidean_dist(z, episode_matrix)
                else:
                    raise NotImplementedError
            else:
                z = self.dropout(z)
                z = episode_classifier(z)

            loss = loss_fn(input=z, target=batch_labels)
            acc = (z.argmax(1) == batch_labels).float().mean()
            loss.backward()
            optimizer.step()
            iter_bar.set_description(f"{loss.item():.3f} | {acc.item():.3f}")

            if summary_writer:
                summary_writer.add_scalar(tag=f'{summary_tag_prefix}_loss', global_step=iteration, scalar_value=loss.item())
                summary_writer.add_scalar(tag=f'{summary_tag_prefix}_acc', global_step=iteration, scalar_value=acc.item())

        # Predict on query
        self.eval()
        if not self.is_pp:
            episode_classifier.eval()

        query_data_list = [{"sentence": sentence, "label": label} for label, sentences in query_data_dict.items() for sentence in sentences]
        query_labels = torch.Tensor([class_to_ix[d['label']] for d in query_data_list]).long().to(device)
        logits = list()
        with torch.no_grad():
            for ix in range(0, len(query_data_list), 16):
                batch = query_data_list[ix:ix + 16]
                batch_sentences = [d['sentence'] for d in batch]
                batch_embeddings = torch.Tensor([sentence_to_embedding_dict[s] for s in batch_sentences]).to(device)
                # z = self.encoder(batch_sentences)
                z = batch_embeddings

                if self.is_pp:
                    if self.metric == "cosine":
                        z = cosine_similarity(z, episode_matrix) * 5
                    elif self.metric == "euclidean":
                        z = -euclidean_dist(z, episode_matrix)
                    else:
                        raise NotImplementedError
                else:
                    z = episode_classifier(z)

                logits.append(z)
        logits = torch.cat(logits, dim=0)
        y_hat = logits.argmax(1)

        y_pred = logits.argmax(1).cpu().detach().numpy()
        probas_pred = logits.cpu().detach().numpy()
        probas_pred = np.exp(probas_pred) / np.exp(probas_pred).sum(1)[:, None]

        y_true = query_labels.cpu().detach().numpy()
        where_ok = np.where(y_pred == y_true)[0]
        import uuid
        tag = str(uuid.uuid4())
        summary_writer.add_text(tag=tag, text_string=json.dumps(ix_to_class, ensure_ascii=False), global_step=0)
        if len(where_ok):
            # Looking for OK but with less confidence (not too easy)
            ok_idx = sorted(where_ok, key=lambda x: probas_pred[x][y_pred[x]])[0]
            ok_sentence = query_data_list[ok_idx]['sentence']
            ok_prediction = ix_to_class[y_pred[ok_idx]]
            ok_label = query_data_list[ok_idx]['label']
            summary_writer.add_text(
                tag=tag,
                text_string=json.dumps({
                    "sentence": ok_sentence,
                    "true_label": ok_label,
                    "predicted_label": ok_prediction,
                    "p": probas_pred[ok_idx].tolist(),
                }),
                global_step=1)

        where_ko = np.where(y_pred != y_true)[0]
        if len(where_ko):
            # Looking for KO but with most confidence
            ko_idx = sorted(where_ko, key=lambda x: probas_pred[x][y_pred[x]], reverse=True)[0]
            ko_sentence = query_data_list[ko_idx]['sentence']
            ko_prediction = ix_to_class[y_pred[ko_idx]]
            ko_label = query_data_list[ko_idx]['label']
            summary_writer.add_text(
                tag=tag,
                text_string=json.dumps({
                    "sentence": ko_sentence,
                    "true_label": ko_label,
                    "predicted_label": ko_prediction,
                    "p": probas_pred[ko_idx].tolist()
                }),
                global_step=2)

        loss = loss_fn(input=logits, target=query_labels)
        acc = (y_hat == query_labels).float().mean()

        return {
            "loss": loss.item(),
            "acc": acc.item()
        }

    def test_model(
            self,
            data_dict: Dict[str, List[str]],
            n_support: int,
            n_classes: int,
            n_episodes=600,
            summary_writer: SummaryWriter = None,
            n_test_iter: int = 100,
            test_batch_size: int = 4
    ):
        test_metrics = list()

        # Freeze encoder
        self.encoder.eval()
        logger.info("Embedding sentences...")
        sentences_to_embed = [s for label, sentences in data_dict.items() for s in sentences]
        sentence_to_embedding_dict = {s: self.encoder.forward([s]).cpu().detach().numpy().squeeze() for s in tqdm.tqdm(sentences_to_embed)}

        for episode in tqdm.tqdm(range(n_episodes)):
            episode_classes = np.random.choice(list(data_dict.keys()), size=n_classes, replace=False)
            episode_query_data_dict = dict()
            episode_support_data_dict = dict()

            for episode_class in episode_classes:
                random.shuffle(data_dict[episode_class])
                episode_support_data_dict[episode_class] = data_dict[episode_class][:n_support]
                episode_query_data_dict[episode_class] = data_dict[episode_class][n_support:]

            episode_metrics = self.test_one_episode(
                support_data_dict=episode_support_data_dict,
                query_data_dict=episode_query_data_dict,
                n_iter=n_test_iter,
                batch_size=test_batch_size,
                sentence_to_embedding_dict=sentence_to_embedding_dict,
                summary_writer=summary_writer
            )
            logger.info(f"Episode metrics: {episode_metrics}")
            test_metrics.append(episode_metrics)
            for metric_name, metric_value in episode_metrics.items():
                summary_writer.add_scalar(tag=metric_name, global_step=episode, scalar_value=metric_value)

        return test_metrics


def run_baseline(
        train_path: str,
        model_name_or_path: str,
        n_support: int,
        n_classes: int,
        valid_path: str = None,
        test_path: str = None,
        output_path: str = f'runs/{now()}',
        n_test_episodes: int = 600,
        log_every: int = 10,
        n_train_epoch: int = 400,
        train_batch_size: int = 16,
        is_pp: bool = False,
        test_batch_size: int = 4,
        n_test_iter: int = 100,
        metric: str = "cosine",
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
            labels_dict[item['label']].append(item["sentence"])
        labels_dict = dict(labels_dict)
        if shuffle:
            for key, val in labels_dict.items():
                random.shuffle(val)
        return labels_dict

    # Load model
    bert = BERTEncoder(model_name_or_path).to(device)
    baseline_net = BaselineNet(encoder=bert, is_pp=is_pp, metric=metric).to(device)

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

        baseline_net.train_model(
            data_dict=train_data_dict,
            summary_writer=train_writer,
            n_epoch=n_train_epoch,
            batch_size=train_batch_size,
            log_every=log_every
        )

        # Validation
        if valid_path:
            validation_metrics = baseline_net.test_model(
                data_dict=valid_data_dict,
                n_support=n_support,
                n_classes=n_classes,
                n_episodes=n_test_episodes,
                summary_writer=valid_writer,
                n_test_iter=n_test_iter,
                test_batch_size=test_batch_size
            )
            with open(os.path.join(output_path, 'validation_metrics.json'), "w") as file:
                json.dump(validation_metrics, file, ensure_ascii=False)
        # Test
        if test_path:
            test_metrics = baseline_net.test_model(
                data_dict=test_data_dict,
                n_support=n_support,
                n_classes=n_classes,
                n_episodes=n_test_episodes,
                summary_writer=test_writer
            )

            with open(os.path.join(output_path, 'test_metrics.json'), "w") as file:
                json.dump(test_metrics, file, ensure_ascii=False)

    else:
        # baseline_net.train_model_ARSC(
        #     train_summary_writer=train_writer,
        #     n_episodes=10,
        #     n_train_iter=20
        # )
        # metrics = baseline_net.test_model_ARSC(
        #     n_iter=n_test_iter,
        #     valid_summary_writer=valid_writer,
        #     test_summary_writer=test_writer
        # )
        metrics = baseline_net.run_ARSC(
            train_summary_writer=train_writer,
            valid_summary_writer=valid_writer,
            test_summary_writer=test_writer,
            n_episodes=1000,
            train_eval_every=50,
            n_train_iter=50,
            n_test_iter=200,
            test_eval_every=25,
            data_path=data_path
        )
        with open(os.path.join(output_path, 'baseline_metrics.json'), "w") as file:
            json.dump(metrics, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--test-path", type=str, default=None, help="Path to testing data")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data (ARSC only)")

    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-test-episodes", type=int, default=600, help="Number of episodes during evaluation (valid, test)")
    parser.add_argument("--n-train-epoch", type=int, default=400, help="Number of epoch during training")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Batch size used during training")
    parser.add_argument("--n-test-iter", type=int, default=100, help="Number of training iterations during testing episodes")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Batch size used during training")

    # Baseline++
    parser.add_argument("--pp", default=False, action="store_true", help="Boolean to use the ++ baseline model")
    parser.add_argument("--metric", default="cosine", type=str, help="Which metric to use in baseline++", choices=("euclidean", "cosine"))
    # ARSC data
    parser.add_argument("--arsc-format", default=False, action="store_true", help="Using ARSC few-shot format")

    args = parser.parse_args()
    logger.info(f"Received args:{args}")

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.train_path, args.valid_path, args.test_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_baseline(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        data_path=args.data_path,
        output_path=args.output_path,

        model_name_or_path=args.model_name_or_path,

        n_support=args.n_support,
        n_classes=args.n_classes,
        n_test_episodes=args.n_test_episodes,
        log_every=args.log_every,
        n_train_epoch=args.n_train_epoch,
        train_batch_size=args.train_batch_size,
        is_pp=args.pp,

        test_batch_size=args.test_batch_size,
        n_test_iter=args.n_test_iter,
        metric=args.metric,
        arsc_format=args.arsc_format
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
