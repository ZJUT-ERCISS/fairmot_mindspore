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
"""Focal loss."""

from mindspore import nn, ops

from src.utils.class_factory import ClassFactory, ModuleType

# focal loss: afa=2, beta=4


@ClassFactory.register(ModuleType.LOSS)
class FocalLoss(nn.Cell):
    """
    nn.Cell warpper for focal loss.

    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples.
            Default: 4.

    Returns:
        Tensor, focal loss.
    """

    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.log = ops.Log()
        self.pow = ops.Pow()
        self.sum = ops.ReduceSum()

    def construct(self, pred, gt):
        """Construct method"""
        pos_inds = ops.Select()(ops.Equal()(gt, 1.0), ops.Fill()(ops.DType()(gt), ops.Shape()(gt), 1.0),
                                ops.Fill()(ops.DType()(gt),
                                           ops.Shape()(gt),
                                           0.0))
        neg_inds = ops.Select()(ops.Less()(gt, 1.0), ops.Fill()(ops.DType()(gt), ops.Shape()(gt), 1.0),
                                ops.Fill()(ops.DType()(gt),
                                           ops.Shape()(gt),
                                           0.0))

        neg_weights = self.pow(1 - gt, self.beta)  # beta=4
        # afa=2
        pos_loss = self.log(pred) * self.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = self.log(1 - pred) * self.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = self.sum(pos_inds, ())
        num_pos = ops.Select()(ops.Equal()(num_pos, 0.0), ops.Fill()(ops.DType()(num_pos), ops.Shape()(num_pos), 1.0),
                               num_pos)

        pos_loss = self.sum(pos_loss, ())
        neg_loss = self.sum(neg_loss, ())
        loss = - (pos_loss + neg_loss) / num_pos
        return loss
