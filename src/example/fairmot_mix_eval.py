# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" MobileNetV2 eval script. """

import os
import argparse
import numpy as np

from mindspore import context, load_checkpoint, load_param_into_net
import motmetrics as mm

from src.models.fairmot import FairmotDla34
from src.utils.post_process.eval_utils.eval_seq import eval_seq
from src.utils.post_process.eval_utils.load_images import LoadImages
from src.utils.post_process.tracker.multitracker import JDETracker
from src.utils.post_process.tracking_utils.evaluation import Evaluator
from src.utils.post_process.infer_net import InferNet, WithInferNetCell


def fairmot_dla34_eval(args_opt):
    """fairmot dla34 eval."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)

    # Create model.
    network = FairmotDla34(down_ratio=args_opt.down_ratio)
    if args_opt.ckpt_path:
        param_dict = load_checkpoint(args_opt.ckpt_path)
        load_param_into_net(network, param_dict)
    infer_net = InferNet()
    net = WithInferNetCell(network, infer_net)
    net.set_train(False)

    # Calculate eval results.
    accs = []
    timer_avgs, timer_calls = [], []
    # start_idx = {'MOT17-02-SDP': 300,
    #                  'MOT17-04-SDP': 525,
    #                  'MOT17-05-SDP': 418,
    #                  'MOT17-09-SDP': 262,
    #                  'MOT17-10-SDP': 327,
    #                  'MOT17-11-SDP': 450,
    #                  'MOT17-14-SDP': 375}
    for seq in args_opt.seqs:
        # dataloader = LoadImages(os.path.join(args_opt.data_root, seq, 'img1'),
        #                         0,
        #                         (1088, 608))
        dataloader = LoadImages(os.path.join(args_opt.data_root, seq, 'img1'),
                                0,
                                (1088, 608))
        tracker = JDETracker(args_opt.conf_thres,
                             args_opt.track_buffer,
                             args_opt.K,
                             args_opt.num_classes,
                             frame_rate=30)
        seq_save_dir = os.path.join(args_opt.output_dir, seq)
        result_filename = os.path.join(seq_save_dir, 'eval_result.txt')
        _, ta, tc = eval_seq(net,
                             dataloader,
                             tracker,
                             args_opt.down_ratio,
                             args_opt.min_box_area,
                             args_opt.data_type,
                             result_filename,
                             start_id=0,
                             save_dir=seq_save_dir,
                             show_image=False)
        evaluator = Evaluator(args_opt.data_root, seq, args_opt.data_type)
        accs.append(evaluator.eval_file(result_filename))
    timer_avgs = np.asarray(ta)
    timer_calls = np.asarray(tc)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    print('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, args_opt.seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileNetV2 eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_root', required=True, default=None, help='Path to sequences of video frames.')
    parser.add_argument('--ckpt_path', default=None, help='Path to trained model weights.')
    parser.add_argument('--seqs_str', type=str, default='mot17', choices=['mot16', 'mot17'], help="Name of dataset.")
    parser.add_argument('--output_dir', type=str, default='output', help="Relative path to result files.")
    parser.add_argument('--save_videos', type=bool, default=True, help='Whether to save videos.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')

    parser.add_argument('--reg_loss', type=str, default='l1', help='Number of batch size.')
    parser.add_argument('--hm_weight', type=int, default=1, help='Loss weight for keypoint heatmaps.')
    parser.add_argument('--wh_weight', type=int, default=0.1, help='Loss weight for bounding box size.')
    parser.add_argument('--off_weight', type=int, default=1, help='Loss weight for keypoint local offsets.')
    parser.add_argument('--reg_offset', type=bool, default=True, help='Whether to use regress local offset.')
    parser.add_argument('--reid_dim', type=int, default=128, help='Feature embed dim.')
    parser.add_argument('--nID', type=int, default=14455, help='Totoal number of identities in dataset.')

    parser.add_argument('--down_ratio', type=int, default=4, help='Totoal number of identities in dataset.')
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence thresh for tracking')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--K', type=int, default=500, help='max number of output objects.')
    parser.add_argument('--num_classes', type=int, default=1, help='num_classes')
    args = parser.parse_known_args()[0]
    # data_root = '/home/publicfile/dataset/MOT17/images/train'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.seqs_str == 'mot17':
        args.seqs = ['MOT17-02-SDP',
                     'MOT17-04-SDP',
                     'MOT17-05-SDP',
                     'MOT17-09-SDP',
                     'MOT17-10-SDP',
                     'MOT17-13-SDP']
        args.data_type = 'mot'
    if args.seqs_str == 'mot16':
        args.seqs = ['MOT16-01',
                     'MOT16-03',
                     'MOT16-06',
                     'MOT16-07',
                     'MOT16-08',
                     'MOT16-12',
                     'MOT16-14']
        args.data_type = 'mot'
    # args.seqs = ['MOT20-01',
    #              'MOT20-02',
    #              'MOT20-03',
    #              'MOT20-05']
    # args.data_type = 'mot'
    fairmot_dla34_eval(args)
