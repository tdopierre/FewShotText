#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -p GPU
#SBATCH -J protonet
#SBATCH -o protonet.log
#SBATCH --gres=gpu:1
#SBATCH --exclude=calcul-gpu-lahc-2

cd $HOME/Projects/FewShotText
source .venv/bin/activate
source .envrc
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPU Devices:"
gpustat

PYTHONPATH=. python models/proto/protonet.py $@
