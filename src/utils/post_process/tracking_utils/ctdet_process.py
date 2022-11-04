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
centernet dets post process
"""

import numpy as np
from ..image_utils import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
    """
    dets: batch x max_dets x dim
    return: 1-based class det dict
    """
    result = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            idxes = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, idxes, :4].astype(np.float32),
                dets[i, idxes, 4:5].astype(np.float32)], axis=1).tolist()
        result.append(top_preds)
    return result
