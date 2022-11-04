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
"""MindSpore Vision Video tracking infer script."""

import os
import numpy as np
from mindspore import context, load_checkpoint, load_param_into_net

from src.utils.config import parse_args, Config
from src.models import build_model
from src.utils.post_process.infer_net import InferNet, WithInferNetCell
from src.utils.post_process.eval_utils.eval_seq import eval_seq
from src.utils.post_process.eval_utils.load_images import LoadImages
from src.utils.post_process.tracker.multitracker import JDETracker
from src.utils.post_process.tracking_utils.evaluation import Evaluator
from src.utils.post_process.infer_net import InferNet, WithInferNetCell

import motmetrics as mm

def eval_tracking(pargs):
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)

    # set network
    network = build_model(config.model)

    # load pretrain model
    param_dict = load_checkpoint(config.infer.ckpt_path)
    load_param_into_net(network, param_dict)

    # init the whole Model
    infer_net = InferNet()
    net = WithInferNetCell(network, infer_net)
    net.set_train(False)

    # Calculate eval results.
    accs = []
    timer_avgs, timer_calls = [], []
    if config.infer.data_seqs == "mot16":
        eval_seqs = ['MOT16-01',
                     'MOT16-03',
                     'MOT16-06',
                     'MOT16-07',
                     'MOT16-08',
                     'MOT16-12',
                     'MOT16-14']
    elif config.infer.data_seqs == "mot17":
        eval_seqs = ['MOT17-01-SDP',
                     'MOT17-03-SDP',
                     'MOT17-06-SDP',
                     'MOT17-07-SDP',
                     'MOT17-08-SDP',
                     'MOT17-14-SDP']
    elif config.infer.data_seqs == "mot20":
        eval_seqs = ['MOT20-04',
                     'MOT20-06',
                     'MOT20-07',
                     'MOT20-08']
    if not os.path.exists(config.infer.output_dir):
        os.mkdir(config.infer.output_dir)
    for seq in eval_seqs:
        dataloader = LoadImages(os.path.join(config.infer.data_root, seq, 'img1'),
                                0,
                                (1088, 608))
        tracker = JDETracker(config.infer.conf_thres,
                             config.infer.track_buffer,
                             config.infer.max_objs,
                             config.infer.num_classes,
                             frame_rate=30)
        seq_save_dir = os.path.join(config.infer.output_dir, seq)
        result_filename = os.path.join(seq_save_dir, 'eval_result.txt')
        _, ta, tc = eval_seq(net,
                             dataloader,
                             tracker,
                             config.infer.down_ratio,
                             config.infer.min_box_area,
                             config.infer.data_type,
                             result_filename,
                             start_id=0,
                             save_dir=seq_save_dir,
                             show_image=False)
        evaluator = Evaluator(config.infer.data_root, seq, config.infer.data_type)
        accs.append(evaluator.eval_file(result_filename))
    if config.infer.save_videos == True:
        output_video_path = os.path.join(config.infer.output_dir, '{}.mp4'.format(seq))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(config.infer.output_dir, output_video_path)
        os.system(cmd_str)
    timer_avgs = np.asarray(ta)
    timer_calls = np.asarray(tc)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    print('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, config.infer.data_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


if __name__ == '__main__':
    args = parse_args()
    eval_tracking(args)
