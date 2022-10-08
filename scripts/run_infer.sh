#!/bin/bash
# bash run_infer.sh [CKPT_PATH] [DATA_ROOT] [SAVE_VIDEOS] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
if [ $# -lt 4 ]; then
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [CKPT_PATH] [DATA_ROOT] [SAVE_VIDEOS] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
    [CKPT_PATH] is path to trained model weight file,
    [DATA_ROOT] is directory to evaluation data,
    [SAVE_VIDEOS] whether to save results into videos,
    [VISIBLE_DEVICES] chooses which gpu device will be used during evaluation."
exit 1
fi

export PYTHONPATH=$PWD
echo "export PYTHONPATH=$PWD"

if [ ! -d "output" ]; then
    mkdir output
fi

CKPT_PATH=$1
DATA_ROOT=$2
SAVE_VIDEOS=$3
export CUDA_VISIBLE_DEVICES="$4"
python src/example/fairmot_mix_infer.py --ckpt_path ${CKPT_PATH} --data_root ${DATA_ROOT} --save_videos ${SAVE_VIDEOS} > output/infer.log 2>&1 &
