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
gaussian utils
"""

import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Set label value of gt bbox within gaussian radius.
    Details of why using `gaussian_radius` can be found in paper:
    https://arxiv.org/abs/1808.01244.

    Args:
        det_size (tuple[int]): Size of ground truth bounding box.
        min_overlap (float): Threshold of iou which is calculated by gt bbox and
            bbox that is within radius. Default: 0.7.

    Returns:
        Minimum radius that meet the overlap condition.

    """
    h, w = det_size

    a1 = 1
    a2 = 4
    a3 = 4 * min_overlap

    b1 = (h + w)
    b2 = 2 * (h + w)
    b3 = -2 * min_overlap * (h + w)

    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    c2 = c2 = (1 - min_overlap) * w * h
    c3 = c3 = (min_overlap - 1) * w * h

    sqrt1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    sqrt2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    sqrt3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)

    radius1 = (b1 + sqrt1) / 2  # denominator should be 2*a
    radius2 = (b2 + sqrt2) / 2
    radius3 = (b3 + sqrt3) / 2

    return min(radius1, radius2, radius3)


def gaussian2d(shape, sigma=1):
    """
    Gaussian2d heatmap.

    Args:
        shape (tuple[int]): x, y radius of gaussian dustribution.
        sigma (int, float): Standard deviation of gaussian dustribution. Default: 1.

    Returns:
        gaussian heatmap mask.
    """
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draw umich gaussian, apply gaussian distribution to heatmap.

    Args:
        heatmap (numpy.ndarray): Heatmap.
        center (sequence[int]): Center of gaussian mask.
        radius (int, float): Radius of gaussian mask.
        k (int, float): Multiplier for gaussian mask values.

    Returns:
        heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    h, w = heatmap.shape[0:2]

    x, y = int(center[0]), int(center[1])

    top, bottom = min(y, radius), min(h - y, radius + 1)
    left, right = min(x, radius), min(w - x, radius + 1)

    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_msra_gaussian(heatmap, center, sigma):
    """
    Draw msra gaussian, apply gaussian distribution to heatmap.

    Args:
        heatmap (numpy.ndarray): Heatmap.
        center (sequence[int]): Center of gaussian mask.
        sigma (int, float): Standard deviation of gaussian dustribution.

    Returns:
        heatmap.
    """
    temp_size = sigma * 3
    mu_y = int(center[1] + 0.5)
    mu_x = int(center[0] + 0.5)
    width, height = heatmap.shape[0], heatmap.shape[1]
    br = [int(mu_x + temp_size + 1), int(mu_y + temp_size + 1)]
    ul = [int(mu_x - temp_size), int(mu_y - temp_size)]
    if ul[0] >= height or ul[1] >= width or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * temp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_y = max(0, -ul[1]), min(br[1], width) - ul[1]
    g_x = max(0, -ul[0]), min(br[0], height) - ul[0]
    img_y = max(0, ul[1]), min(br[1], width)
    img_x = max(0, ul[0]), min(br[0], height)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap