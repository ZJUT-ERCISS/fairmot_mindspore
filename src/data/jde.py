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
""" JDE dataset. """

import os.path as osp
import json
import numpy as np


from src.data.meta import Dataset, ParseDataset
from src.utils.class_factory import ClassFactory, ModuleType
from src.data.transforms.jde_load import JDELoad

__all__ = ["MixJDE", "JDE", "ParseJDE"]


@ClassFactory.register(ModuleType.DATASET)
class MixJDE(Dataset):
    """Multi-dataset based on jde datasets.

    Args:
        data_json (str): Path to a json file that have the path to files that have the path to video frames.
        split (str): The dataset split supports "train", or "test". Default: "train".
        batch_size (int): Batch size of dataset. Default:1.
        repeat_num (int): The repeat num of dataset. Default:1.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.

    """

    def __init__(self,
                 data_json,
                 split="train",
                 batch_size=1,
                 repeat_num=1,
                 transform=None,
                 shuffle=None,
                 resize=224,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 schema_json=None,
                 trans_record=None,):

        with open(data_json, 'r', encoding='utf-8') as f:
            data = f.read()
            data_dict = json.loads(data)
        self.seq_root = data_dict['seq_root']
        self.data_root = data_dict['data_root']
        self.batch_size = batch_size
        self.repeat_num = repeat_num
        self.images_path = []
        self.images_label = []
        for dname, dpath in data_dict[split].items():
            seq_path = osp.join(self.seq_root, dpath)
            print(f'Loading {dname}...')
            images_path_tmp, images_label_tmp = ParseJDE(seq_path).parse_dataset(self.data_root)
            self.images_path.extend(images_path_tmp)
            self.images_label.extend(images_label_tmp)
            print(f'{dname} loaded.')
            load_data = self.read_dataset
            self.output_columns = ['imgs', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'ids']
            self.input_columns = ['imgs', 'label']

        super(MixJDE, self).__init__(path=self.data_root,
                                     split=split,
                                     load_data=load_data,
                                     transform=transform,
                                     target_transform=None,
                                     batch_size=batch_size,
                                     repeat_num=repeat_num,
                                     resize=resize,
                                     shuffle=shuffle,
                                     num_parallel_workers=num_parallel_workers,
                                     num_shards=num_shards,
                                     shard_id=shard_id,
                                     columns_list=self.input_columns,
                                     schema_json=schema_json,
                                     trans_record=trans_record)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise ValueError("JDE dataset has no label.")

    def download_dataset(self):
        """Download the JDE data if it doesn't exist already."""
        raise ValueError("JDE dataset download is not supported.")

    def read_dataset(self, *args):
        if not args:
            return self.images_path, self.images_label
        return self.images_path[args[0]], self.images_label[args[0]]
        # return np.fromfile(self.images_path[args[0]], dtype="int8"), self.images_label[args[0]]

    def default_transform(self):
        """Default data augmentation."""
        has_outside_points = True
        if "MOT20" in self.images_path[0]:
            has_outside_points = False
        trans = [JDELoad((1088, 608), clip_outside_points=has_outside_points)]
        return trans

    def pipelines(self):
        """Data augmentation."""
        if not self.dataset:
            raise ValueError("dataset is None")

        trans = self.transform if self.transform else self.default_transform()
        self.dataset = self.dataset.map(operations=trans,
                                        input_columns=self.input_columns,
                                        output_columns=self.output_columns,
                                        column_order=self.output_columns,
                                        num_parallel_workers=self.num_parallel_workers)


@ClassFactory.register(ModuleType.DATASET)
class JDE(Dataset):
    """`
    Args:
        seq_path (str): Path to a file that have the path to video frames.
        data_root (str): Path to
        split (str): The dataset split supports "train", or "test". Default: "train".
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq (int): The number of frames of captured video. Default: 16.
        seq_mode (str): The way of capture video frames,"part"), "discrete", or "align" fetch. Default: "align".
        align (boolean): The video contains multiple actions.Default: False.
        batch_size (int): Batch size of dataset. Default:1.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.

    Examples:
        >>> from mindvision.msvideo.dataset.jde import JDE
        >>> dataset = JDE("./data", "train")
        >>> dataset = dataset.run()

    The davis structure of JDE dataset looks like:

    .. code-block::
        .
        |-mot16
            |-- test        //contains 7 videos.
            |   |-- MOT16-01
            |       |-- det
            |           |-- det.txt   // annotation file
            |       |-- img1
            |           |-- 000001.jpg   // a frame of image in a video.
            |           |-- 000002.jpg   // a frame of image in a video.
            |           |...
            |       |-- seqinfo.ini     // video information file.
            |   |-- MOT16-03
            |       |-- det
            |           |-- det.txt   // annotation file
            |       |-- img1
            |           |-- 000001.jpg   // a frame of image in a video.
            |           |-- 000002.jpg   // a frame of image in a video.
            |           |...
            |       |-- seqinfo.ini     // video information file.
            |   ...
            |-- train       //contains 7 videos.
            |   |-- MOT16-02
            |       |-- det
            |           |-- det.txt   // annotation file
            |       |-- gt
            |           |-- gt.txt   // annotation file
            |       |-- img1
            |           |-- 000001.jpg   // a frame of image in a video.
            |           |-- 000002.jpg   // a frame of image in a video.
            |           |...
            |       |-- seqinfo.ini     // video information file.
            |   |-- MOT16-04
            |       |-- det
            |           |-- det.txt   // annotation file
            |       |-- gt
            |           |-- gt.txt   // annotation file
            |       |-- img1
            |           |-- 000001.jpg   // a frame of image in a video.
            |           |-- 000002.jpg   // a frame of image in a video.
            |           |...
            |       |-- seqinfo.ini     // video information file.

    """

    def __init__(self,
                 seq_path,
                 data_root,
                 split="train",
                 transform=None,
                 batch_size=1,
                 repeat_num=1,
                 resize=224,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 schema_json=None,
                 trans_record=None,
                 ):
        self.images_path, self.images_label = ParseJDE(seq_path).parse_dataset(data_root)
        load_data = self.read_dataset
        self.output_columns = ['imgs', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'ids']
        self.input_columns = ['imgs', 'label']

        super(JDE, self).__init__(path=data_root,
                                  split=split,
                                  load_data=load_data,
                                  transform=transform,
                                  target_transform=None,
                                  batch_size=batch_size,
                                  repeat_num=repeat_num,
                                  resize=resize,
                                  shuffle=shuffle,
                                  num_parallel_workers=num_parallel_workers,
                                  num_shards=num_shards,
                                  shard_id=shard_id,
                                  columns_list=self.input_columns,
                                  schema_json=schema_json,
                                  trans_record=trans_record)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise ValueError("JDE dataset has no label.")

    def download_dataset(self):
        """Download the JDE data if it doesn't exist already."""
        raise ValueError("JDE dataset download is not supported.")

    def read_dataset(self, *args):
        if not args:
            return self.images_path, self.images_label
        return self.images_path[args[0]], self.images_label[args[0]]
        # return np.fromfile(self.images_path[args[0]], dtype="int8"), self.images_label[args[0]]

    def default_transform(self):
        """Default data augmentation."""
        has_outside_points = True
        if "MOT20" in self.images_path[0]:
            has_outside_points = False
        trans = [JDELoad((1088, 608), clip_outside_points=has_outside_points)]
        return trans

    def pipelines(self):
        """Data augmentation."""
        if not self.dataset:
            raise ValueError("dataset is None")

        trans = self.transform if self.transform else self.default_transform()
        self.dataset = self.dataset.map(operations=trans,
                                        input_columns=self.input_columns,
                                        output_columns=self.output_columns,
                                        column_order=self.output_columns,
                                        num_parallel_workers=self.num_parallel_workers)


class ParseJDE(ParseDataset):
    """
    Parse JDE dataset.
    """

    def parse_dataset(self, *args):
        """Traverse the JDE dataset file to get the path and label."""

        parse_jde = ParseJDE(self.path)  # data list path
        data_root = args[0]
        with open(parse_jde.path, 'r', encoding='utf-8') as file:
            img_paths = file.readlines()
            img_paths = [x.replace('\n', '') for x in img_paths]
            img_paths = list(filter(lambda x: len(x) > 0, img_paths))
            img_paths = [osp.join(data_root, p) for p in img_paths]

        label_paths = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                       for x in img_paths]
        labels = [np.loadtxt(lp, dtype=np.float32).reshape(-1, 6)
                  for lp in label_paths]

        return img_paths, labels
