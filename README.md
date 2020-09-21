# FewShotText
Library of various Few-Shot Learning frameworks for text classification

## Fine-tuning BERT on the MLM task
```bash
model_name=bert-base-cased
block_size=256
dataset=OOS
output_dir=transformer_models/${dataset}/fine-tuned

python scripts_transformers/run_language_modeling.py \
        --model_name_or_path ${model_name} \
        --output_dir ${output_dir} \
        --mlm \
        --do_train \
        --train_data_file data/${dataset}/full/full-train.txt  \
        --do_eval \
        --eval_data_file data/${dataset}/full/full-test.txt \
        --overwrite_output_dir \
        --evaluate_during_training \
        --logging_steps=1000 \
        --line_by_line \
        --logging_dir ${output_dir} \
        --block_size ${block_size} \
        --save_steps=1000 \
        --num_train_epochs 20 \
        --save_total_limit 20 \
        --seed 42
```

## Training a few-shot model
To run the paper's experiments, simply use the ```utils/scripts/runner.sh``` file. 
