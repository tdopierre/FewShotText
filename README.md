# FewShotText
This repository contains code for the paper [A Neural Few-Shot Text Classification Reality Check](https://arxiv.org/abs/2101.12073)

## Environment setup
```bash
# Create environment
python3 -m virtualenv .venv --python=python3.6

# Install environment
.venv/bin/pip install -r requirements.txt

# Activate environment
source .venv/bin/activate
```

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

## Reference
If you use the data or codes in this repository, please cite our paper:
```bash
@article{dopierre2021neural,
    title={A Neural Few-Shot Text Classification Reality Check},
    author={Dopierre, Thomas and Gravier, Christophe and Logerais, Wilfried},
    journal={arXiv preprint arXiv:2101.12073},
    year={2021}
}
```