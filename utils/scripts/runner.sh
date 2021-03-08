#!/usr/bin/env bash

for n_class in 5 2 3 4; do
    for seed in 1 2 3 4 5; do
#    for shots in 5 10; do
        for shots in 5; do
            for dataset in OOS TREC28 Liu; do
                TRAIN_FILE="data/${dataset}/train.jsonl"
                VALID_FILE="data/${dataset}/valid.jsonl"
                TEST_FILE="data/${dataset}/test.jsonl"
                bert_model_name_or_path="/home/dot10713/Projects/UDA_pytorch/transformer_models/${dataset}/fine-tuned"
                OUTPUT_ROOT="runs/${dataset}/${n_class}C_${shots}K/seed${seed}"

                # Baseline
                OUT_PATH="${OUTPUT_ROOT}/baseline"

                if [[ -d "${OUT_PATH}" ]]; then
                    echo "${OUT_PATH} already exists. Skipping."
                else
                    mkdir -p ${OUT_PATH}
                    LOGS_PATH="${OUT_PATH}/training.log"

                    sbatch \
                        -n 1 \
                        -p GPU \
                        --gres=gpu:1 --nice=1800 \
                        --exclude=calcul-gpu-lahc-2 \
                        -J ${OUT_PATH} \
                        -o ${LOGS_PATH} \
                        models/bert_baseline/baseline.sh \
                        --train-path ${TRAIN_FILE} \
                        --valid-path ${VALID_FILE} \
                        --test-path ${TEST_FILE} \
                        --model-name-or-path "${bert_model_name_or_path}" \
                        --n-test-episodes 600 \
                        --n-support ${shots} \
                        --n-classes ${n_class} \
                        --n-train-epoch 10 \
                        --output-path "${OUT_PATH}/output" --seed ${seed}
                fi

                # Induction
                OUT_PATH="${OUTPUT_ROOT}/induction"
                if [[ -d "${OUT_PATH}" ]]; then
                    echo "${OUT_PATH} already exists. Skipping."
                else
                    mkdir -p ${OUT_PATH}
                    LOGS_PATH="${OUT_PATH}/training.log"

                    sbatch \
                        -n 1 \
                        -p GPU \
                        --gres=gpu:1 --nice=1800 \
                        --exclude=calcul-gpu-lahc-2 \
                        -J "${OUT_PATH}" \
                        -o ${LOGS_PATH} \
                        models/induction/inductionnet.sh \
                        --train-path ${TRAIN_FILE} \
                        --valid-path ${VALID_FILE} \
                        --test-path ${TEST_FILE} \
                        --model-name-or-path "${bert_model_name_or_path}" \
                        --n-test-episodes 600 \
                        --n-support ${shots} \
                        --n-query 5 \
                        --n-classes ${n_class} \
                        --max-iter 10000 \
                        --evaluate-every 100 \
                        --output-path "${OUT_PATH}/output" \
                        --ntl-n-slices 100 \
                        --n-routing-iter 3 \
                        --early-stop 25 --seed ${seed}
                fi

                # Relation-base
                OUT_PATH="${OUTPUT_ROOT}/relation-base"
                if [[ -d "${OUT_PATH}" ]]; then
                    echo "${OUT_PATH} already exists. Skipping."
                else
                    mkdir -p ${OUT_PATH}
                    LOGS_PATH="${OUT_PATH}/training.log"

                    sbatch \
                        -n 1 \
                        -p GPU \
                        --gres=gpu:1 --nice=1800 \
                        --exclude=calcul-gpu-lahc-2 \
                        -J "${OUT_PATH}" \
                        -o ${LOGS_PATH} \
                        models/relation/relationnet.sh \
                        --train-path ${TRAIN_FILE} \
                        --valid-path ${VALID_FILE} \
                        --test-path ${TEST_FILE} \
                        --model-name-or-path "${bert_model_name_or_path}" \
                        --n-test-episodes 600 \
                        --n-support ${shots} \
                        --n-query 5 \
                        --n-classes ${n_class} \
                        --max-iter 10000 \
                        --evaluate-every 100 \
                        --output-path "${OUT_PATH}/output" \
                        --relation-module-type "base" \
                        --early-stop 25 --seed ${seed}
                fi

                # Relation-ntl
                OUT_PATH="${OUTPUT_ROOT}/relation-ntl"
                if [[ -d "${OUT_PATH}" ]]; then
                    echo "${OUT_PATH} already exists. Skipping."
                else
                    mkdir -p ${OUT_PATH}
                    LOGS_PATH="${OUT_PATH}/training.log"

                    sbatch \
                        -n 1 \
                        -p GPU \
                        --gres=gpu:1 --nice=1800 \
                        --exclude=calcul-gpu-lahc-2 \
                        -J "${OUT_PATH}" \
                        -o "${LOGS_PATH}" \
                        models/relation/relationnet.sh \
                        --train-path ${TRAIN_FILE} \
                        --valid-path ${VALID_FILE} \
                        --test-path ${TEST_FILE} \
                        --model-name-or-path "${bert_model_name_or_path}" \
                        --n-test-episodes 600 \
                        --n-support ${shots} \
                        --n-query 5 \
                        --n-classes ${n_class} \
                        --max-iter 10000 \
                        --evaluate-every 100 \
                        --output-path "${OUT_PATH}/output" \
                        --relation-module-type "ntl" \
                        --early-stop 25 --seed ${seed}
                fi

                for metric in euclidean cosine; do
                    # Baseline++
                    OUT_PATH="${OUTPUT_ROOT}/baseline++_${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        sbatch \
                            -n 1 \
                            -p GPU \
                            --gres=gpu:1 --nice=1800 \
                            --exclude=calcul-gpu-lahc-2 \
                            -J ${OUT_PATH} \
                            -o ${LOGS_PATH} \
                            models/bert_baseline/baseline.sh \
                            --train-path ${TRAIN_FILE} \
                            --valid-path ${VALID_FILE} \
                            --test-path ${TEST_FILE} \
                            --model-name-or-path "${bert_model_name_or_path}" \
                            --n-test-episodes 600 \
                            --n-support ${shots} \
                            --n-classes ${n_class} \
                            --n-train-epoch 10 \
                            --output-path "${OUT_PATH}/output" \
                            --pp \
                            --metric ${metric} --seed ${seed}
                    fi

                    # Matching
                    OUT_PATH="${OUTPUT_ROOT}/matching-${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        sbatch \
                            -n 1 \
                            -p GPU \
                            --gres=gpu:1 --nice=1800 \
                            --exclude=calcul-gpu-lahc-2 \
                            -J ${OUT_PATH} \
                            -o ${LOGS_PATH} \
                            models/matching/matchingnet.sh \
                            --train-path ${TRAIN_FILE} \
                            --valid-path ${VALID_FILE} \
                            --test-path ${TEST_FILE} \
                            --model-name-or-path "${bert_model_name_or_path}" \
                            --n-test-episodes 600 \
                            --n-support ${shots} \
                            --n-query 5 \
                            --n-classes ${n_class} \
                            --max-iter 10000 \
                            --evaluate-every 100 \
                            --output-path "${OUT_PATH}/output" \
                            --metric ${metric} \
                            --early-stop 25 --seed ${seed}
                    fi

                    # Proto
                    OUT_PATH="${OUTPUT_ROOT}/proto-${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        sbatch \
                            -n 1 \
                            -p GPU \
                            --gres=gpu:1 --nice=1800 \
                            --exclude=calcul-gpu-lahc-2 \
                            -J ${OUT_PATH} \
                            -o ${LOGS_PATH} \
                            models/proto/protonet.sh \
                            --train-path ${TRAIN_FILE} \
                            --valid-path ${VALID_FILE} \
                            --test-path ${TEST_FILE} \
                            --model-name-or-path "${bert_model_name_or_path}" \
                            --n-test-episodes 600 \
                            --n-support ${shots} \
                            --n-query 5 \
                            --n-classes ${n_class} \
                            --max-iter 10000 \
                            --evaluate-every 100 \
                            --output-path "${OUT_PATH}/output" \
                            --metric ${metric} \
                            --early-stop 25 --seed ${seed}
                    fi

                    # Proto++
                    OUT_PATH="${OUTPUT_ROOT}/proto++-${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        if [[ "${dataset}" == "Liu" ]]; then
                            n_unlabeled=10
                        else
                            n_unlabeled=20
                        fi
                        sbatch \
                            -n 1 \
                            -p GPU \
                            --gres=gpu:titanxt:1 --nice=1800 \
                            --exclude=calcul-gpu-lahc-2 \
                            -J ${OUT_PATH} \
                            -o ${LOGS_PATH} \
                            models/proto/protonet.sh \
                            --train-path ${TRAIN_FILE} \
                            --valid-path ${VALID_FILE} \
                            --test-path ${TEST_FILE} \
                            --model-name-or-path "${bert_model_name_or_path}" \
                            --n-test-episodes 600 \
                            --n-support ${shots} \
                            --n-query 5 \
                            --n-classes ${n_class} \
                            --max-iter 10000 \
                            --evaluate-every 100 \
                            --output-path "${OUT_PATH}/output" \
                            --metric ${metric} \
                            --early-stop 25 \
                            --n-unlabeled ${n_unlabeled} --seed ${seed}
                    fi
                done
            done
        done
    done
done
