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
"""FairMOT classification head."""

from mindspore import nn
from mindspore.common.initializer import Zero, Constant
from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.LAYER)
class FairMOTSingleHead(nn.Cell):
    """Simple convolutional head, two conv2d layers will be created if head_conv > 0,
       else there is only one conv2d layer.

    Args:
        in_channel(int): Channel size of input feature.
        head_conv(int): Channel size between two conv2d layers, there will be only one conv2d layer
            if head_conv equals 0. Default: 0.
        classes(int): Number of classes, channel size of output tensor.
        kernel_size(Union[int, tuple]): The kernel size of first conv2d layer.
        bias_init(Union[Tensor, str, Initializer, numbers.Number]): Bias initialization of last conv2d layer.
            The input value is the same as `mindspore.common.initializer.initializer`.

    Returns:
        Tensor, the classification result.
    """

    def __init__(self, in_channel, head_conv=0, classes=100, kernel_size=3, bias_init=Zero()):
        super(FairMOTSingleHead, self).__init__()
        if head_conv > 0:
            self.fc = nn.SequentialCell(
                [nn.Conv2d(in_channel, head_conv, kernel_size=3, has_bias=True),
                 nn.ReLU(),
                 nn.Conv2d(head_conv, classes, kernel_size=kernel_size, bias_init=bias_init, has_bias=True)]
            )
        else:
            self.fc = nn.Conv2d(in_channel, classes, kernel_size=kernel_size, bias_init=bias_init,
                                has_bias=True)

    def construct(self, y):
        """Input feature and get classification result using conv2d layers."""
        return self.fc(y)


@ClassFactory.register(ModuleType.LAYER)
class FairMOTMultiHead(nn.Cell):
    """Fairmot net multi-conv head, the combination of single heads.

    Args:
        heads(dict): A dict contains name and output dimension of heads, the name is the key, and output
                     dimension is the value. For fairmot, it must have 'hm', 'wh', 'id', 'reg' heads.
        in_channel(int): Channel size of input feature.
        head_conv(int): Channel size between two conv2d layers, there will be only one conv2d layer
            if head_conv equals 0. Default: 0.
        kernel_size(Union[int, tuple]): The kernel size of first conv2d layer.
        bias_init(Union[Tensor, str, Initializer, numbers.Number]): Bias initialization of last conv2d layer.
            The input value is the same as `mindspore.common.initializer.initializer`.

    Returns:
        Tensor, the multi-head classification results.
    """

    def __init__(self, heads, in_channel, head_conv=0, kernel_size=3):
        super(FairMOTMultiHead, self).__init__()
        self.hm_fc = FairMOTSingleHead(in_channel, head_conv, heads['hm'], kernel_size, bias_init=Constant(-2.19))
        self.wh_fc = FairMOTSingleHead(in_channel, head_conv, heads['wh'], kernel_size)
        self.id_fc = FairMOTSingleHead(in_channel, head_conv, heads['feature_id'], kernel_size)
        self.reg_fc = FairMOTSingleHead(in_channel, head_conv, heads['reg'], kernel_size)

    def construct(self, y):
        """Input feature and get classification result using conv2d layers."""
        hm = self.hm_fc(y)
        wh = self.wh_fc(y)
        feature_id = self.id_fc(y)
        reg = self.reg_fc(y)
        feature = {"hm": hm, "feature_id": feature_id, "wh": wh, "reg": reg}
        return feature
