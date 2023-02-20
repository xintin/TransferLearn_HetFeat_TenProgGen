#!/bin/bash 
#COBALT -t 1:00:00 -n 1 -q single-gpu
#COBALT -A datascience
#COBALT -O xgb_train_by_task
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

time python3 train_model.py --models xgb  --split-scheme by_task --use-gpu
# mv xgb.pkl ../out_datasets/xgb_train_by_task.pkl
# mv xgb_mlp.pkl ../out_datasets/tmp_xgb_train_by_task.pkl

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "Job End Time: ${end_fmt}"

