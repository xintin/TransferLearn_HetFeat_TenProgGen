#!/bin/bash 
#COBALT -t 2:00:00 -n 1 -q bigmem
#COBALT -A datascience
#COBALT -O xbg_tune_within_task_resnet50
#COBALT --attrs filesystems=home,theta-fs0,grand,eagle


. /etc/profile.d/z00_lmod.sh

module load conda/pytorch
conda activate tvm_build

cd /lus/grand/projects/datascience/sraskar/projects/tvm_100/tenset/scripts
CUDA_VISIBLE_DEVICES=1

export TVM_HOME=/lus/theta-fs0/projects/datascience/gverma/tenset
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

HOST=$(hostname)
echo "hostname: $HOST"

WORKDIR=$PWD
echo "DIR: $WORKDIR"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "Job Start Time: ${start_fmt}"

echo "==== Batch Size 1 ===="
time python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb --load-model /lus/theta-fs0/projects/datascience/gverma/tenset/scripts/A100_models_800/xgb_gpu_800_within_task.pkl --batch-size 1 --target "cuda"

echo "==== Batch Size 2 ===="
time python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb --load-model /lus/theta-fs0/projects/datascience/gverma/tenset/scripts/A100_models_800/xgb_gpu_800_within_task.pkl --batch-size 2 --target "cuda"

echo "==== Batch Size 4 ===="
time python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb --load-model /lus/theta-fs0/projects/datascience/gverma/tenset/scripts/A100_models_800/xgb_gpu_800_within_task.pkl --batch-size 4 --target "cuda"

echo "==== Batch Size 8 ===="
time python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb --load-model /lus/theta-fs0/projects/datascience/gverma/tenset/scripts/A100_models_800/xgb_gpu_800_within_task.pkl --batch-size 8 --target "cuda"

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "Job End Time: ${end_fmt}"

