#!/usr/bin/env bash

for n_class in 5 2 3 4; do
    for seed in 1 2 3 4 5; do
        for shots in 5; do
            for dataset in OOS TREC28 Liu; do

                OUTPUT_ROOT="runs/${dataset}/${n_class}C_${shots}K/seed${seed}"
                data_params="
                    --train-path data/${dataset}/train.jsonl
                    --valid-path data/${dataset}/valid.jsonl
                    --test-path data/${dataset}/test.jsonl"

                baseline_params="
                    --n-classes ${n_class}
                    --n-support ${shots}
                    --n-test-episodes 600"

                few_shot_params="
                    --n-classes ${n_class}
                    --n-support ${shots}
                    --n-query 5
                    --n-test-episodes 600"

                model_params="
                    --model-name-or-path transformer_models/${dataset}/fine-tuned"


                baseline_training_params="
                    --n-train-epoch 10
                    --seed ${seed}"

                few_shot_training_params="
                    --max-iter 10000
                    --evaluate-every 100
                    --early-stop 25
                    --seed ${seed}"

                # Baseline
                OUT_PATH="${OUTPUT_ROOT}/baseline"
                if [[ -d "${OUT_PATH}" ]]; then
                    echo "${OUT_PATH} already exists. Skipping."
                else
                    mkdir -p ${OUT_PATH}
                    LOGS_PATH="${OUT_PATH}/training.log"

                    ./models/bert_baseline/baseline.sh \
                        $(echo ${data_params}) \
                        $(echo ${baseline_params}) \
                        $(echo ${model_params})  \
                        $(echo ${baseline_training_params}) \
                        --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                fi

                # Induction Network
                OUT_PATH="${OUTPUT_ROOT}/induction"
                if [[ -d "${OUT_PATH}" ]]; then
                    echo "${OUT_PATH} already exists. Skipping."
                else
                    mkdir -p ${OUT_PATH}
                    LOGS_PATH="${OUT_PATH}/training.log"

                    ./models/induction/inductionnet.sh \
                        $(echo ${data_params}) \
                        $(echo ${few_shot_params}) \
                        $(echo ${model_params})  \
                        $(echo ${few_shot_training_params}) \
                        --ntl-n-slices 100 \
                        --n-routing-iter 3  \
                        --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                fi

                # Relation Network
                for relation_module_type in base ntl; do

                    OUT_PATH="${OUTPUT_ROOT}/relation-${relation_module_type}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        ./models/relation/relationnet.sh \
                            $(echo ${data_params}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${model_params})  \
                            $(echo ${few_shot_training_params}) \
                            --relation-module-type "${relation_module_type}" \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}

                    fi
                done


                for metric in euclidean cosine; do
                    # Baseline++
                    OUT_PATH="${OUTPUT_ROOT}/baseline++_${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        ./models/bert_baseline/baseline.sh \
                            $(echo ${data_params}) \
                            $(echo ${baseline_params}) \
                            $(echo ${model_params})  \
                            $(echo ${baseline_training_params}) \
                            --pp --metric "${metric}" \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # Matching Network
                    OUT_PATH="${OUTPUT_ROOT}/matching-${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        ./models/matching/matchingnet.sh \
                            $(echo ${data_params}) \
                            $(echo ${model_params})  \
                            $(echo ${few_shot_params}) \
                            $(echo ${few_shot_training_params}) \
                            --metric ${metric} \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # Prototypical Network
                    OUT_PATH="${OUTPUT_ROOT}/proto-${metric}"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        ./models/proto/protonet.sh \
                            $(echo ${data_params}) \
                            $(echo ${model_params})  \
                            $(echo ${few_shot_params}) \
                            $(echo ${few_shot_training_params}) \
                            --metric ${metric} \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
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
                        ./models/proto/protonet.sh \
                            $(echo ${data_params}) \
                            $(echo ${model_params})  \
                            $(echo ${few_shot_params}) \
                            $(echo ${few_shot_training_params}) \
                            --metric ${metric} \
                            --n-unlabeled ${n_unlabeled} \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi
                done
            done
        done
    done
done
