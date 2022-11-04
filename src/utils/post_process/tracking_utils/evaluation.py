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
"""
Evaluator for fairmot.
"""
import copy
import os.path as osp
import numpy as np
import motmetrics as mm
from ..tracking_utils.io import read_results, unzip_objs

mm.lap.default_solver = 'lap'


class Evaluator:
    """
    Evaluate tracking results.
    """

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def eval_file(self, filename):
        """evaluate file"""
        self.reset_accumulator()

        res_frame_dict = read_results(filename, self.data_type, is_gt=False)
        # frames = sorted(list(set(self.gt_frame_dict.keys()) | set(res_frame_dict.keys())))
        frames = sorted(list(set(res_frame_dict.keys())))
        for f_id in frames:
            trk_objs = res_frame_dict.get(f_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(f_id, trk_tlwhs, trk_ids, ret_events=False)

        return self.acc

    def reset_accumulator(self):
        """reset accumulator"""
        self.acc = mm.MOTAccumulator(auto_id=True)

    def load_annotations(self):
        """load annotations"""
        assert self.data_type == 'mot', "input data_type should be `mot`."

        gt_fname = osp.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_fname, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_fname, self.data_type, is_ignore=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, ret_events=False):
        """eval frame"""
        # results
        trk_ids = np.copy(trk_ids)
        trk_tlwhs = np.copy(trk_tlwhs)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep_mask = np.ones(len(trk_tlwhs), dtype=bool)
        iou_dis = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if np.size(iou_dis):
            match_rows, match_cols = mm.lap.linear_sum_assignment(iou_dis)
            match_rows, match_cols = map(lambda a: np.asarray(a, dtype=int), [match_rows, match_cols])
            match_ious = iou_dis[match_rows, match_cols]

            match_cols = np.asarray(match_cols, dtype=int)
            match_cols = match_cols[np.logical_not(np.isnan(match_ious))]
            keep_mask[match_cols] = False
            trk_ids = trk_ids[keep_mask]
            trk_tlwhs = trk_tlwhs[keep_mask]

        # match_rows, match_cols = mm.lap.linear_sum_assignment(iou_dis)
        # match_rows, match_cols = map(lambda a: np.asarray(a, dtype=int), [match_rows, match_cols])
        # match_ious = iou_dis[match_rows, match_cols]

        # match_cols = np.asarray(match_cols, dtype=int)
        # match_cols = match_cols[np.logical_not(np.isnan(match_ious))]
        # keep[match_cols] = False
        # trk_tlwhs = trk_tlwhs[keep]
        # trk_ids = trk_ids[keep]

        # get distance matrix
        iou_dis = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_dis)

        # only supported by https://github.com/longcw/py-motmetrics

        if ret_events and iou_dis.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    @staticmethod
    def get_summary(accs, names, metrics=None):
        """summary results"""
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        """save summary"""
        import pandas
        writer = pandas.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
