echo "========================================================================"
echo "Please run the script as: "
echo "bash run_standalone_train_ascend.sh DEVICE_ID DATA_CFG(options) LOAD_PRE_MODEL(options)"
echo "For example: bash run_standalone_train_ascend.sh 0 ./dataset/ ./dla34.ckpt"
echo "It is better to use the absolute path."
echo "========================================================================"
export DEVICE_ID=$1
DATA_CFG=$2
LOAD_PRE_MODEL=$3
echo "start training for device $DEVICE_ID"
python -u ../train.py --device Ascend --id ${DEVICE_ID} --data_cfg ${DATA_CFG} --load_pre_model ${LOAD_PRE_MODEL} --is_modelarts False > train${DEVICE_ID}.log 2>&1 &
echo "finish"