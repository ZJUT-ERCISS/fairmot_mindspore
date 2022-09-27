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
"""JDE data load transforms functions."""

import copy
import math
import random
import cv2
import numpy as np

import mindspore.dataset.transforms.py_transforms as trans
from src.utils.class_factory import ClassFactory, ModuleType
from src.utils.gaussian import draw_msra_gaussian, draw_umich_gaussian, gaussian_radius


@ClassFactory.register(ModuleType.PIPELINE)
class JDELoad(trans.PyTensorOperation):
    """
    Load jde dataset and augment images.

    Args:
        size (tuple[int]): Size of output images. Default: (1088, 608).
        max_objs (int): Maximum number of objects in an image. Default: 500.
        ltrb (bool): Format of label.
            If True, label should be in format: `(left, top, right, bottom)`.
            If False, label should be in format: `(x, y, w, h)`.
            Default: True.

    Inputs:
        img_path (str): Path to image.
        labels0 (sequence): normalized xywh labels.

    Returns:
        image, heatmap, reg_mask, identity feature, bbox size, regress local offset, index
    """

    def __init__(self, size=(1088, 608), max_objs=500, ltrb=True):
        self.size = tuple(size)
        self.width = size[0]
        self.height = size[1]
        self.max_objs = max_objs
        self.ltrb = ltrb
        self.mse_loss = True
        self.num_classes = 1

    def __call__(self, img_path, labels0):

        # read img
        img_path = bytes.decode(img_path.tostring())
        # print(type(img_path), img_path)
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
        # random color adjust
        img = self.random_color_adjust(img)
        h, w, _ = img.shape
        img, ratio, padw, padh = self.letterbox(img, height=self.height, width=self.width)
        # Normalized xywh to pixel xyxy format
        labels = labels0.copy()
        labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
        labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
        labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
        labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        # random affine
        img, labels, _ = self.random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))
        # xyxy to xywh
        labels[:, 2:6] = self.xyxy2xywh(labels[:, 2:6].copy())
        labels[:, 2] /= self.width
        labels[:, 3] /= self.height
        labels[:, 4] /= self.width
        labels[:, 5] /= self.height
        # random left-right flip
        lr_flip = True
        if lr_flip & (random.random() > 0.5):
            img = np.fliplr(img)
            labels[:, 2] = 1 - labels[:, 2]
        # rescale to [0.0, 1.0]
        img = np.array(img, dtype=np.float32) / 255
        # HWC to CHW
        img = img.transpose((2, 0, 1))

        output_h = img.shape[1] // 4  # down_ratio = 4
        output_w = img.shape[2] // 4
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int32)
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)
        ids = np.zeros((self.max_objs,), dtype=np.int32)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.mse_loss else draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.mse_loss else radius
                # radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                        bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_xy
        # ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids,
        # 'bbox': bbox_xys}
        return img, hm, reg_mask, ind, wh, reg, ids

    def random_color_adjust(self, img):
        """
        Random color and saturation adjust.

        Args:
            img (numpy.ndarray): Image data.

        Returns:
            Augmented image.

        """
        fraction = 0.50
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = img_hsv[:, :, 1].astype(np.float32)
        val = img_hsv[:, :, 2].astype(np.float32)

        a = (random.random() * 2 - 1) * fraction + 1
        sat *= a
        if a > 1:
            np.clip(sat, a_min=0, a_max=255, out=sat)

        a = (random.random() * 2 - 1) * fraction + 1
        val *= a
        if a > 1:
            np.clip(val, a_min=0, a_max=255, out=val)

        img_hsv[:, :, 1] = sat.astype(np.uint8)
        img_hsv[:, :, 2] = val.astype(np.uint8)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)
        return img

    def letterbox(self, img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):
        """
        Resize a rectangular image to a padded rectangular

        Args:
            img (numpy.ndarray): Image that will be resized.
            height (int): Height of resized image.
            width (int): Width of resized image.
            color (tuple[float]): Color of padded area in resized image.

        Returns:
            image, resize ratio, padding width, padding height
        """
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

    def random_affine(self, img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                      border_value=(127.5, 127.5, 127.5)):
        """
        Apply Random affine transformation to the input image.

        targets (sequence, optional): Label of image. It should have at least 6 numbers in length.
            Default: None.
        degrees (sequence): Range of the rotation degrees. Default: (-10, 10).
        translate (sequence): Sequence (tx_min, tx_max, ty_min, ty_max) of minimum/maximum translation
            in x(horizontal) and y(vertical) directions. The horizontal and vertical shift is selected
            randomly from the range: (tx_min*width, tx_max*width) and (ty_min*height, ty_max*height),
            respectively. Default: (0.1, 0.1).
        scale (sequence): Scaling factor interval, which must be non negative. Default: (0.9, 1,1).
        shear (sequence): Range of shear factor, which must be positive, a shear parallel to X axis in the
            range of (shear[0], shear[1]) and a shear parallel to Y axis in the range of (shear[0], shear[1])
            is applied. Default: (-2, 2).
        border_value (Union[int, tuple[int]], optional): Optional fill_value to fill the area outside the transform
            in the output image. There must be three elements in tuple and the value of single element is [0, 255].
            Default: (127.5, 127.5, 127.5).

        Returns:
            Warped image, and warped labels, affine matrix if `targets` is not None.
        """

        border = 0  # width of added border (optional)
        height = img.shape[0]
        width = img.shape[1]

        # Rotation and Scale
        rotation = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        rotation[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        translation = np.eye(3)
        translation[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
        translation[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        sh = np.eye(3)
        sh[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        sh[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        mat = np.matmul(sh, np.matmul(translation, rotation))  # Combined rotation matrix. ORDER IS IMPORTANT HERE!
        imw = cv2.warpPerspective(img, mat, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=border_value)  # BGR order borderValue

        # Return warped points also
        if targets is not None:
            if np.shape(targets)[0] > 0:
                n = targets.shape[0]
                points = targets[:, 2:6].copy()
                area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = np.matmul(xy, mat.T)[:, :2].reshape(n, 8)
                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # apply angle-based reduction
                radians = a * math.pi / 180
                reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                x = (xy[:, 2] + xy[:, 0]) / 2
                y = (xy[:, 3] + xy[:, 1]) / 2
                w = (xy[:, 2] - xy[:, 0]) * reduction
                h = (xy[:, 3] - xy[:, 1]) * reduction
                xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                # reject warped points outside of image
                np.clip(xy[:, 0], 0, width, out=xy[:, 0])
                np.clip(xy[:, 2], 0, width, out=xy[:, 2])
                np.clip(xy[:, 1], 0, height, out=xy[:, 1])
                np.clip(xy[:, 3], 0, height, out=xy[:, 3])
                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

                targets = targets[i]
                targets[:, 2:6] = xy[i]

            return imw, targets, mat
        return imw

    def xyxy2xywh(self, x):
        # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        y = np.zeros(x.shape)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y
