#!/bin/bash 
#COBALT -t 4:00:00 -n 1 -q gpu_a40
#COBALT -O xgb_trimmed_inception_v3
#COBALT --attrs filesystems=home,theta-fs0,grand,eagle


source /home/sraskar/miniconda3/etc/profile.d/conda.sh
export PATH=/home/sraskar/miniconda3/bin:$PATH

module use /soft/modulefiles/

module load llvm/release-11.1.0
# conda init bash
conda activate tvm_build

cd /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/tenset/scripts
CUDA_VISIBLE_DEVICES=1

export TVM_HOME=/gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

HOST=$(hostname)
echo "hostname: $HOST"

WORKDIR=$PWD
echo "DIR: $WORKDIR"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "Job Start Time: ${start_fmt}"

echo "======== By Target ========"

echo "===== Without Transfer Tuning ====="
echo "=== Batch Size 1 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 1 --target "cuda"
echo "=== Batch Size 2 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 2 --target "cuda"
echo "=== Batch Size 3 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 4 --target "cuda"
echo "=== Batch Size 4 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 8 --target "cuda"

echo "===== With Transfer Tuning ====="
echo "=== Batch Size 1 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 1 --transfer-tune --target "cuda"
echo "=== Batch Size 2 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 2 --transfer-tune --target "cuda"
echo "=== Batch Size 4 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 4 --transfer-tune --target "cuda"
echo "=== Batch Size 8 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytarget_trimmed_gpu.pkl --batch-size 8 --transfer-tune --target "cuda"


echo "======== By Task ========"

echo "===== Without Transfer Tuning ====="
echo "=== Batch Size 1 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 1 --target "cuda"
echo "=== Batch Size 2 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 2 --target "cuda"
echo "=== Batch Size 3 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 4 --target "cuda"
echo "=== Batch Size 4 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 8 --target "cuda"

echo "===== With Transfer Tuning ====="
echo "=== Batch Size 1 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 1 --transfer-tune --target "cuda"
echo "=== Batch Size 2 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 2 --transfer-tune --target "cuda"
echo "=== Batch Size 4 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 4 --transfer-tune --target "cuda"
echo "=== Batch Size 8 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_bytask_trimmed_task_gpu.pkl --batch-size 8 --transfer-tune --target "cuda"

echo "======== Within Task ========"

echo "===== Without Transfer Tuning ====="
echo "=== Batch Size 1 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 1 --target "cuda"
echo "=== Batch Size 2 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 2 --target "cuda"
echo "=== Batch Size 3 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 4 --target "cuda"
echo "=== Batch Size 4 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 8 --target "cuda"

echo "===== With Transfer Tuning ====="
echo "=== Batch Size 1 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 1 --transfer-tune --target "cuda"
echo "=== Batch Size 2 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 2 --transfer-tune --target "cuda"
echo "=== Batch Size 4 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 4 --transfer-tune --target "cuda"
echo "=== Batch Size 8 ==="
time python3 tune_network.py --network inception_v3 --n-trials 100 --cost-model xgb --load-model /gpfs/jlse-fs0/users/sraskar/projects/tvm_a40/efficient_transfer_tuning_tenset/rtx_xenon/models/xgb_within_trimmed_task_gpu.pkl --batch-size 8 --transfer-tune --target "cuda"

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "Job End Time: ${end_fmt}"