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
img utils
"""
import numpy as np
import cv2


def plot_tracking(image, tlwhs, obj_ids, frame_id=0, fps=0., ids2=None):
    """plot tracking"""
    im = np.ascontiguousarray(np.copy(image))

    text_scale = max(1, image.shape[1] / 1500.)
    line_thickness = max(1, int(image.shape[1] / 450.))
    num = len(tlwhs)
    cv2.putText(im, f'frame: {frame_id} fps: {fps:.2f} num: {num}', (0, int(15 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i in range(num):
        x, y, w, h = tlwhs[i]
        integerbox = tuple(map(int, (x, y, x + w, y + h)))
        obj_id = int(obj_ids[i])
        id_text = f'{obj_id}'
        if ids2 is not None:
            ids = int(ids2[i])
            id_text = f'{obj_id}, {ids}'
        # _line_thickness = 1 if obj_id <= 0 else line_thickness
        idx = abs(obj_id) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        cv2.rectangle(im, integerbox[0:2], integerbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (integerbox[0], integerbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale,
                    (0, 0, 255), thickness=2)
    return im
