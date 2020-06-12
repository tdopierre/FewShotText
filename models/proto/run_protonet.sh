#!/usr/bin/env bash

# Testing cluster
if [[ "$1" == "slow" ]]; then
    echo "This is a slow run"
fi

# Running
for cv in 01 02 03 04 05 06 07 08 09 10; do
    for dataset in snips OOS r52 Liu TREC28 Lara; do
#    for dataset in Liu; do
#    for dataset in TREC28 Liu; do

        # Vanilla Protonet
        TRAIN_IN_PATH=${HOME}/Projects/query-clustering/data/datasets/${dataset}/few-shot_final/${cv}/train.jsonl
        TEST_OUT_PATH=${HOME}/Projects/query-clustering/data/proto/${dataset}/${cv}/proto-embeddings.pkl
        TEST_IN_PATH=${HOME}/Projects/query-clustering/data/datasets/${dataset}/full/input
        MODEL_NAME_OR_PATH=${HOME}/Projects/UDA_pytorch/transformer_models/${dataset}/fine-tuned

        LOGS_PATH="${HOME}/logs/proto/${dataset}/${cv}/training.log"
        mkdir -p "${HOME}/logs/proto/${dataset}/${cv}"

        if [[ -f "${TEST_OUT_PATH}" ]]; then
            echo "${TEST_OUT_PATH} already exists. Skipping."
        else
            sbatch \
                -J "proto/${dataset}/${cv}" \
                -o ${LOGS_PATH} \
                nlu/models/protonet.sh \
                --train-file-path ${TRAIN_IN_PATH} \
                --test-input-file-path ${TEST_IN_PATH} \
                --test-output-file-path ${TEST_OUT_PATH} \
                --model-name-or-path ${MODEL_NAME_OR_PATH}
        fi

        # Proto-Net++ (refined prototypes with soft k-means)
        TRAIN_IN_PATH=${HOME}/Projects/query-clustering/data/datasets/${dataset}/few-shot_final/${cv}/train.jsonl
        TEST_OUT_PATH=${HOME}/Projects/query-clustering/data/proto/${dataset}/${cv}/proto++-embeddings.pkl
        TEST_IN_PATH=${HOME}/Projects/query-clustering/data/datasets/${dataset}/full/input
        MODEL_NAME_OR_PATH=${HOME}/Projects/UDA_pytorch/transformer_models/${dataset}/fine-tuned

        LOGS_PATH="${HOME}/logs/proto++/${dataset}/${cv}/training.log"
        mkdir -p "${HOME}/logs/proto++/${dataset}/${cv}"

        if [[ -f "${TEST_OUT_PATH}" ]]; then
            echo "${TEST_OUT_PATH} already exists. Skipping."
        else
            sbatch \
                -J "proto++/${dataset}/${cv}" \
                -o ${LOGS_PATH} \
                nlu/models/protonet.sh \
                --train-file-path ${TRAIN_IN_PATH} \
                --test-input-file-path ${TEST_IN_PATH} \
                --test-output-file-path ${TEST_OUT_PATH} \
                --model-name-or-path ${MODEL_NAME_OR_PATH} \
                --refined
        fi

        if [[ "$1" == "slow" ]]; then
            sleep 5
        fi
    done
done