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
""" Fairmot load imgs."""


import glob
import os.path as osp
import cv2
import numpy as np


def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    """resize a rectangular image to a padded rectangular"""
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(width * 1.0 / shape[1], height * 1.0 / shape[0])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    l, r = round(dw - 0.1), round(dw + 0.1)
    t, b = round(dh - 0.1), round(dh + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


class LoadImages:
    """
    Load images for inference.

    Args:
        path (str): Path to dataset directory.
        start_idx (int): First reading index in dataset. Default: 0.
        img_size (tuple[int]): Shape of resized frame image. Default: (1088, 608).

    Returns:
        image path, processed image and origin image.
    """

    def __init__(self, path, start_idx=0, img_size=(1088, 608)):
        if osp.isdir(path):
            formats = ['.jpg', '.jpeg', '.png', '.tif']
            self.filepaths = sorted(glob.glob(f'{path}/*.*'))
            self.filepaths = list(filter(lambda x: osp.splitext(x)[1].lower() in formats, self.filepaths))
        elif osp.isfile(path):
            self.filepaths = [path]
        self.width = img_size[0]
        self.height = img_size[1]

        self.start_idx = start_idx
        self.index = start_idx

        self.nframe = len(self.filepaths) - self.start_idx  # number of image filepaths
        assert self.nframe > 0, f'No images found in {path}'

    def __iter__(self):
        self.index = self.start_idx-1
        return self

    def __next__(self):
        self.index += 1
        if (self.index + self.start_idx) == self.nframe:
            raise StopIteration
        img_path = self.filepaths[self.index]

        # Read image
        image0 = cv2.imread(img_path)  # BGR
        assert image0 is not None, f'Failed to load {img_path}'

        # Padded resize
        image, _, _, _ = letterbox(image0, height=self.height, width=self.width)

        # Normalize RGB
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0

        return img_path, image, image0

    def __getitem__(self, idx):
        idx = (idx + self.start_idx) % self.nframe
        img_path = self.filepaths[idx]

        # Read image
        image0 = cv2.imread(img_path)  # BGR
        assert image0 is not None, f'Failed to load {img_path}'

        # Padded resize
        image, _, _, _ = letterbox(image0, height=self.height, width=self.width)

        # Normalize RGB
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0

        return img_path, image, image0

    def __len__(self):
        return self.nframe
