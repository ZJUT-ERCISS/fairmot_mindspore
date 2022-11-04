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
""" Fairmot eval seq."""

import os
import cv2
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from ..tracking_utils.timer import Timer
from ..tracking_utils import visualization as vis


def write_results(filename, results, data_type='mot'):
    """
    Write eval results.

    Args:
        filename (str): File path where save the tracking results.
        results (list): Tracking results.
        data_type (str): Type of dataset, can be 'mot' or 'kitti'. Default: 'mot'.
    Returns:
        None.
    """
    if data_type == 'mot':
        label_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        label_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(f"{data_type} data type is not supported.")

    with open(filename, 'w') as sf:
        for frame, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                label = label_format.format(frame=frame, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                sf.write(label)
    print('save results to %s', filename)


def eval_seq(net,
             dataloader,
             tracker,
             down_ratio,
             min_box_area,
             data_type,
             result_filename,
             start_id=0,
             save_dir=None,
             show_image=True):
    """
    Tracking objects and evaluate tracking results.

    Args:
        net (mindspore.nn.Cell): Trained tracking network.
        dataloader (Iterable): Dataloader, read frames from dataset.
        tracker (object): Tracker object, it processes detection results and
            transfer it into tracking results.
        down_ratio (int): The ratio of resolution between origin frame and pre-processed frames.
        min_box_area (float): The threshold for filtering the detection bboxes,
            making sure that bboxes are big enough.
        data_type (str): Type of dataset, can be 'mot' or 'kitti'. Default: 'mot'.
        result_filename (str): File path where save the tracking results.
        start_id (int): Index of first frame. Default: 0.
        save_dir (optional[str]): Directory where save frames with bbox. If 'None', frames will not
            be saved. Default: None.
        show_image (bool): Whether to show images on the screen. Default: True.

    Returns:
        None.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    timer = Timer()
    results = []
    frame_id = start_id
    # for path, img, img0 in dataloader:
    for _, img, img0 in dataloader:
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.avg_time)))
        # run tracking
        timer.tic()
        blob = np.expand_dims(img, 0)
        blob = Tensor(blob, mstype.float32)
        # img0 = Tensor(img0, mstype.float32)
        height, width = img0.shape[0], img0.shape[1]
        inp_height, inp_width = [blob.shape[2], blob.shape[3]]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta_data = {'c': c, 's': s, 'out_height': inp_height // down_ratio,
                     'out_width': inp_width // down_ratio}
        id_feature, dets = net(blob)
        online_targets = tracker.update(id_feature.asnumpy(), dets, meta_data)
        online_tlwhs = []
        online_ids = []
        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.avg_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    write_results(result_filename, results, data_type)
    return frame_id, timer.avg_time, timer.calls
