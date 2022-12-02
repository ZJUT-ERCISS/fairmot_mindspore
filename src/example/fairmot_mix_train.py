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

import argparse
from mindspore import nn, Tensor, context, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.communication.management import init, get_rank, get_group_size

from src.utils.check_param import Validator, Rel
from src.loss import CenterNetMultiPoseLoss
from src.schedule import dynamic_lr
from src.data import MixJDE
from src.data.transforms.jde_load import JDELoad
from src.models.fairmot import FairmotDla34
from src.loss.tracking_losscell import TrackingLossCell


def fairmot_dla34_train(args_opt):
    """fairmot train."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)

    # run distribute
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            init()
        else:
            init("nccl")
        print("###run_distribute###")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          parameter_broadcast=True)
        ckpt_save_dir = args_opt.ckpt_save_dir + "ckpt_" + str(rank_id) + "/"
        dataset = MixJDE(args_opt.data_json,
                         batch_size=args_opt.batch_size,
                         num_parallel_workers=args_opt.num_parallel_workers,
                         shuffle=True,
                         num_shards=device_num,
                         shard_id=rank_id,
                         repeat_num=args_opt.repeat_num)
    else:
        ckpt_save_dir = args_opt.ckpt_save_dir
        dataset = MixJDE(args_opt.data_json,
                         batch_size=args_opt.batch_size,
                         num_parallel_workers=args_opt.num_parallel_workers,
                         shuffle=True,
                         repeat_num=args_opt.repeat_num)

    # perpare dataset
    transforms = [JDELoad((1088, 608))]
    dataset.transform = transforms
    dataset_train = dataset.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)
    step_size = dataset_train.get_dataset_size()

    # set network
    network = FairmotDla34(down_ratio=args_opt.down_ratio,
                           head_channel=args_opt.head_channel,
                           hm=args_opt.hm,
                           wh=args_opt.wh,
                           feature_id=args_opt.feature_id,
                           reg=args_opt.reg)

    # set loss
    network_loss = CenterNetMultiPoseLoss(args_opt.reg_loss,
                                          args_opt.hm_weight,
                                          args_opt.wh_weight,
                                          args_opt.off_weight,
                                          args_opt.reg_offset,
                                          args_opt.reid_dim,
                                          args_opt.nID,
                                          args_opt.batch_size)

    # set lr
    steps_per_epoch = int(step_size)
    lr = dynamic_lr(base_lr=args_opt.learning_rate,
                    steps_per_epoch=steps_per_epoch,
                    warmup_steps=args_opt.warmup_steps,
                    warmup_ratio=args_opt.warmup_ratio,
                    epoch_size=args_opt.epoch_size)
    print("steps per epoch:", steps_per_epoch)
    # set optimizer
    network_opt = nn.Adam(network.trainable_params(), learning_rate=Tensor(lr))

    if args_opt.pre_trained:
        # load pretrain model
        param_dict = load_checkpoint(args_opt.ckpt_path)
        load_param_into_net(network, param_dict)

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='fairmot_dla34',
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # init the whole Model
    net_with_loss = TrackingLossCell(network, network_loss)
    fairmot_net = nn.TrainOneStepCell(net_with_loss, network_opt)
    model = Model(fairmot_net)

    # begin to train
    print('[Start training `{}`]'.format('fairmot_dla34'))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=args_opt.dataset_sink_mode)
    print('[End of training `{}`]'.format('fairmot_dla34'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileNetV2 eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    # TODO data_url -> data_url_list
    parser.add_argument('--data_json', required=True, default=None, help='The file that saves location of data.')
    parser.add_argument('--pre_trained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--ckpt_path', default=None, help='Path to trained model weights.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=20, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./fairmot_output", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int, default=6, help='Number of parallel workers.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Distributed parallel training.')

    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch size.')
    parser.add_argument('--epoch_size', type=int, default=30, help='Number of epochs for trainning.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Base learning rate.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Steps for warm up.')
    parser.add_argument('--warmup_ratio', type=float, default=1.0,
                        help='The ratio of learning rate at the end of warming up w.r.t base learning rate.')

    parser.add_argument('--reg_loss', type=str, default='l1', choices=['l1', 'sl1'],
                        help='Regression loss, it can be L1 loss or Smooth L1 loss.')
    parser.add_argument('--hm_weight', type=int, default=1, help='Loss weight for keypoint heatmaps.')
    parser.add_argument('--wh_weight', type=int, default=0.1, help='Loss weight for bounding box size.')
    parser.add_argument('--off_weight', type=int, default=1, help='Loss weight for keypoint local offsets.')
    parser.add_argument('--reg_offset', type=bool, default=True, help='Whether to use regress local offset.')
    parser.add_argument('--reid_dim', type=int, default=128, help='Feature embed dim.')
    parser.add_argument('--nID', type=int, default=14455, help='Totoal number of identities in dataset.')

    parser.add_argument('--down_ratio', type=int, default=4, help='Output stride.')
    parser.add_argument('--head_channel', type=int, default=256,
                        help='Channel of input of second conv2d layer in heads.')
    parser.add_argument('--hm', type=int, default=1, help='Number of heatmap channels.')
    parser.add_argument('--wh', type=int, default=4, help='Dimension of offset and size output.')
    parser.add_argument('--feature_id', type=int, default=128, help='Dimension of identity embedding.')
    parser.add_argument('--reg', type=int, default=2, help='Dimension of local offset.')

    args = parser.parse_known_args()[0]
    fairmot_dla34_train(args)
