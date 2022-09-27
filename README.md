# fairmot_mindspore

- [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Requirements Installation](#requirements-installation)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object tracking. However, little attention has been focused on accomplishing the two tasks in a single network to improve the inference speed. The initial attempts along this path ended up with degraded results mainly because the re-identification branch is not appropriately learned. In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS. This baseline could inspire and help evaluate new ideas in this field. More detail about this model can be found in:

[Paper](https://arxiv.org/abs/2004.01888): Zhang Y, Wang C, Wang X, et al. FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking. 2020.

This repository contains a Mindspore implementation of FairMot based upon original Pytorch implementation (<https://github.com/ifzhang/FairMOT>). The training and validating scripts are also included, and the evaluation results are shown in the [Performance](#performance) section.

# [Model Architecture](#contents)

The overall network architecture of FairMOT is shown below:

[Link](https://arxiv.org/abs/2004.01888)

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: ETH, CalTech, MOT17, CUHK-SYSU, PRW, CityPerson

# [Features](#contents)

# [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Python and dependencies
    - python 3.7.5
    - Cython 0.29.30
    - cython-bbox 0.1.3
    - decord 0.6.0
    - mindspore-gpu 1.6.1
    - ml-collections 0.1.1
    - matplotlib 3.4.1
    - motmetrics 1.2.5
    - numpy 1.21.5
    - Pillow 9.0.1
    - PyYAML 6.0
    - scikit-learn 1.0.2
    - scipy 1.7.3
    - pycocotools 2.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

## [Requirements Installation](#contents)

Some packages in `requirements.txt` need Cython package to be installed first. For this reason, you should use the following commands to install dependencies:

```shell
pip install Cython && pip install -r requirements.txt
```

## [Dataset Preparation](#contents)

FairMot model uses mix dataset to train and validate in this repository. We use the training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) in this part and we call it "MIX". Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16.

**Configure path to dataset root** in `data/data.json` file.

## [Model Checkpoints](#contents)

The pretrain model (DLA-34 backbone) is trained on the the MIX dataset for 30 epochs.
It can be downloaded here: [[fairmot_dla34-30_886.ckpt]]()

## [Running](#contents)

To train the model, run the shell script `scripts/run_standalone_train_ascend.sh` or `scripts/run_standalone_train_gpu.sh` with the format below:

```shell
# standalone training on Ascend
bash scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_CFG(options) LOAD_PRE_MODEL(options)

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [config_file] [pretrained_model]

# distributed training on Ascend
bash scripts/run_distribute_train_ascend.sh RANK_SIZE DATA_CFG(options) LOAD_PRE_MODEL(options)

# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
```

To validate the model, run the shell script `scripts/run_eval.sh` with the format below:

To infer using the model, run the shell script `scripts/run_infer.sh` with the format below:
```shell
bash scripts/run_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

The validate and infer programme will generate pictures with predict bbox, and infer programme can generate video if `save_videos` is True.

# [Script Description](#contents)

## [Training Process](#contents)

### [Training](#contents)

Run `scripts/run_standalone_train_<device>.sh` to train the model standalone. The usage of the script is:

#### Running on Ascend

```shell
bash scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_CFG LOAD_PRE_MODEL
```

For example, you can run the shell command below to launch the training procedure.

```shell
bash scripts/run_standalone_train_ascend.sh 0 ./default_config.yaml ./fairmot_dla34-30_886.ckpt
```

#### Running on GPU

```shell
bash scripts/run_standalone_train_gpu.sh [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the training procedure:

```shell
bash scripts/run_standalone_train_gpu.sh ./default_config.yaml ./fairmot_dla34-30_886.ckpt
```

The model checkpoint will be saved into `./output`.

### [Distributed Training](#contents)

Run `scripts/run_distribute_train_<device>.sh` to train the model distributed. The usage of the script is:

#### Running on Ascend

```shell
bash scripts/run_distribute_train_ascend.sh RANK_SIZE DATA_CFG LOAD_PRE_MODEL
```

For example, you can run the shell command below to launch the distributed training procedure.

```shell
bash scripts/run_distribute_train_ascend.sh 8 ./data.json ./crowdhuman_dla34_ms.ckpt
```

#### Running on GPU

```shell
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the distributed training procedure:

```shell
bash scripts/run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 ./default_config.yaml ./crowdhuman_dla34_ms.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train/tran.log`.

The model checkpoint will be saved into `train/ckpt`.

## [Evaluation Process](#contents)

The evaluation data set was [MOT20](https://motchallenge.net/data/MOT20/)

Run `scripts/run_eval.sh` to evaluate the model. The usage of the script is:

```shell
bash scripts/run_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

For example, you can run the shell command below to launch the validation procedure.

```shell
bash scripts/run_eval.sh GPU ./default_config.yaml ./fairmot-30.ckpt data_path
```

The eval results can be viewed in `eval/eval.log`.

# [Model Description](#contents)

## [Performance](#contents)

### FairMot on MIX dataset with detector

#### Performance parameters

| Parameters          | Ascend Standalone           | Ascend Distributed          | GPU Distributed             |
| ------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       | FairMotNet                  | FairMotNet                  | FairMotNet                  |
| Resource            | Ascend 910                  | 8 Ascend 910 cards          | 8x RTX 3090 24GB            |
| Uploaded Date       | 25/06/2021 (day/month/year) | 25/06/2021 (day/month/year) | 21/02/2021 (day/month/year) |
| MindSpore Version   | 1.2.0                       | 1.2.0                       | 1.5.0                       |
| Training Dataset    | MIX                         | MIX                         | MIX                         |
| Evaluation Dataset  | MOT20                       | MOT20                       | MOT20                       |
| Training Parameters | epoch=30, batch_size=4      | epoch=30, batch_size=4      | epoch=30, batch_size=12     |
| Optimizer           | Adam                        | Adam                        | Adam                        |
| Loss Function       | FocalLoss,RegLoss           | FocalLoss,RegLoss           | FocalLoss,RegLoss           |
| Train Performance   | MOTA:43.8% Prcn:90.9%       | MOTA:42.5% Prcn:91.9%%      | MOTA: 41.2%, Prcn: 90.5%    |
| Speed               | 1pc: 380.528 ms/step        | 8pc: 700.371 ms/step        | 8p: 1047 ms/step            |


# Citation
@misc{MindSpore Vision 2022,
    title={{MindSpore Vision}:MindSpore Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://gitee.com/mindspore/vision}},
    year={2022}
}


Time elapsed: 510.83 seconds, FPS: 6.49

          IDF1   IDP   IDR  Rcll  Prcn   GT MT  PT   ML    FP      FN  IDs     FM  MOTA  MOTP  IDt  IDa IDm

MOT20-01 22.7% 54.0% 14.4% 24.4% 91.9%   90  4  40   46   577   20132  328    513 21.1% 0.237  184  111  17

MOT20-02 17.4% 51.0% 10.5% 19.2% 93.0%  296 11 138  147  2903  163428 1778   3045 16.9% 0.224  824  750  86

MOT20-03  8.8% 47.8%  4.9%  7.7% 75.9%  735  0  86  649  8739  328073 1107   3231  5.0% 0.285  582  472  99

MOT20-05  3.5% 34.2%  1.9%  3.4% 62.3% 1210  0  39 1171 13514  635354 1754   3318  1.1% 0.305  940  828 170

OVERALL   8.0% 45.2%  4.4%  7.7% 78.7% 2331 15 303 2013 25733 1146987 4967  10107  5.2% 0.261 2530 2161 372


# TODO list:
scripts in gpu version

performance in mot17 and mot20

add default config.yaml for training and validation 
