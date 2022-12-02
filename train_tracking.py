# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore Vision Video training script."""

from mindspore import nn, Tensor, context, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.communication.management import init, get_rank, get_group_size

from src.utils.config import parse_args, Config
from src.utils.check_param import Validator, Rel
from src.loss.builder import build_loss
from src.loss.tracking_losscell import TrackingLossCell
from src.schedule.builder import get_lr
from src.optim.builder import build_optimizer
from src.data.builder import build_dataset, build_transforms
from src.models import build_model


def train_tracking(pargs):
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)

    # run distribute
    if config.train.run_distribute:
        if config.device_target == "Ascend":
            init()
        else:
            init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=get_group_size(),
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          parameter_broadcast=True)
        ckpt_save_dir = config.train.ckpt_path + "ckpt_" + str(get_rank()) + "/"
        config.train.data_loader.dataset.num_shards = device_num
        config.train.data_loader.dataset.shard_id = rank_id
        data_set = build_dataset(config.train.data_loader.dataset)
    else:
        ckpt_save_dir = config.train.ckpt_path
        data_set = build_dataset(config.train.data_loader.dataset)

    # perpare dataset
    transforms = build_transforms(config.train.data_loader.map)

    data_set.transform = transforms
    dataset_train = data_set.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)

    # set network
    network = build_model(config.model)

    # set loss
    network_loss = build_loss(config.loss)

    # set lr
    lr_cfg = config.learning_rate
    lr_cfg.steps_per_epoch = dataset_train.get_dataset_size()
    lr = get_lr(lr_cfg)

    # set optimizer
    config.optimizer.params = network.trainable_params()
    config.optimizer.learning_rate = lr
    network_opt = build_optimizer(config.optimizer)

    if config.train.pre_trained:
        # load pretrain model
        param_dict = load_checkpoint(config.train.ckpt_path)
        load_param_into_net(network, param_dict)

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=config.train.save_checkpoint_steps,
        keep_checkpoint_max=config.train.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # init the whole Model
    net_with_loss = TrackingLossCell(network, network_loss)
    fairmot_net = nn.TrainOneStepCell(net_with_loss, network_opt)
    model = Model(fairmot_net)

    # begin to train
    print('[Start training `{}`]'.format(config.model_name))
    print("=" * 80)
    model.train(config.train.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=config.dataset_sink_mode)
    print('[End of training `{}`]'.format(config.model_name))


if __name__ == '__main__':
    args = parse_args()
    train_tracking(args)
