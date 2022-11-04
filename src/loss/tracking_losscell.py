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

from mindspore import nn


class TrackingLossCell(nn.Cell):
    """Cell with loss function.."""

    def __init__(self, net, loss):
        super(TrackingLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, image, hm, reg_mask, ind, wh, reg, ids):
        """Cell with loss function."""
        feature = self._net(image)
        return self._loss(feature, hm, reg_mask, ind, wh, reg, ids)
