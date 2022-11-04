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
"""write and read utils."""

import os
import numpy as np


def write_results(filename, results, data_type):
    """write results"""
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        label_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        label_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(f"{data_type} data type is not supported.")

    with open(filename, 'w') as f:
        for f_id, frame_data in results.items():
            if data_type == 'kitti':
                f_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                label = label_format.format(frame=f_id, id=track_id, x1=x1,
                                            y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(label)
    print('Save results to {}.'.format(filename))


def read_results(filename, data_type, is_gt=False, is_ignore=False):
    """read results"""
    if data_type in ('mot', 'lab'):
        read_func = read_mot_results
    else:
        raise ValueError(f'Data type {data_type} is not supported.')

    return read_func(filename, is_gt, is_ignore)


def read_mot_results(filename, is_gt, is_ignore):
    """read_mot_results"""
    valid_labels = (1,)
    ignore_labels = (2, 7, 8, 12)
    res_dict = {}
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                labellist = line.split(',')
                if len(labellist) < 7:
                    continue
                fid = int(labellist[0])
                if fid < 1:
                    continue
                res_dict.setdefault(fid, [])

                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        mark = int(float(labellist[6]))
                        label = int(float(labellist[7]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        vis_ratio = float(labellist[8])
                        label = int(float(labellist[7]))
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(labellist[6])

                # if box_size > 7000:
                # if box_size <= 7000 or box_size >= 15000:
                # if box_size < 15000:
                # continue

                target_id = int(labellist[1])
                tlwh = tuple(map(float, labellist[2:6]))

                res_dict[fid].append((tlwh, target_id, score))

    return res_dict


def unzip_objs(objs):
    """unzip objs"""
    if objs:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores
