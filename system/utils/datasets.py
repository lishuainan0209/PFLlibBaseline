import json
import os
from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

DATASET_COLLECTION_PATH="../dataset"
class RemoteSensingSegDataset(VisionDataset):
    """
    自定义遥感语义分割数据集（继承VisionDataset）
    适配结构：
    dataset_name/
    ├─ train/
    │  ├─ images/  # 原始遥感图片（1.png, 2.png...）
    │  └─ labels/  # 像素级标签掩码（1.png, 2.png...，与images同名）
    ├─ val/
    │  ├─ images/
    │  └─ labels/
    └─ test/
       ├─ images/
       └─ labels/
    """

    def __init__(
            self,
            dataset_name: str,  # 数据集名称, 与文件夹名称要保持一致
            split: str = "train",  # 选择数据集划分：train/val/test
            transform: Optional[Callable] = None,  # 图片变换（Resize、ToTensor等）
            target_transform: Optional[Callable] = None,  # 标签掩码变换
            img_suffix: str = ".png",  # 图片后缀（适配遥感常见格式：png/tif/jpg）
            data_conf_json: str = None,  # 划分的数据集配置文件
            client_id: int = None,  # 节点名称
    ):
        # 需要用其转换成限向量, 否则__getitem__会出现TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        # ToTensor() 归一化，把 1-255缩放到 0~1 区间,是给图片用的, 不是给标签用的
        # if target_transform is None:
        #     target_transform = transforms.Compose([transforms.ToTensor()])
        # 调用父类VisionDataset初始化（必须）
        super().__init__(os.path.join(DATASET_COLLECTION_PATH, dataset_name), transform=transform, target_transform=target_transform)

        self.split = split
        self.img_suffix = img_suffix
        self.data_conf_json_path = os.path.join(DATASET_COLLECTION_PATH, dataset_name, data_conf_json)
        self.client_id = client_id
        self.img_names = []
        # 1. 定义images和labels的路径
        self.images_dir = os.path.join(DATASET_COLLECTION_PATH, dataset_name, split, "images")
        self.labels_dir = os.path.join(DATASET_COLLECTION_PATH, dataset_name, split, "labels")

        # 校验路径是否存在
        if not os.path.exists(self.images_dir):
            raise ValueError(f"图片目录不存在：{self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise ValueError(f"标签目录不存在：{self.labels_dir}")
        self.load_client_files()

    def load_client_files(self) -> Tuple[List[str], List[str]]:
        # 根据配置文件来划分
        # {
        #       "server":{"test": ["1.png","1.png"],"val": ["1.png","1.png"],},
        #       "client_1":{"train": ["1.png","1.png"],"test": ["1.png","1.png"],"val": ["1.png","1.png"],},
        #       "client_2": {"train": ["1.png", "1.png"], "test": ["1.png", "1.png"], "val": ["1.png", "1.png"], },
        # }
        split_data_conf = None

        # 没有节点名,获取所有该路径下文件名
        if self.client_id is None:
            self.img_names = [f for f in os.listdir(self.images_dir) if f.endswith(self.img_suffix)]
            # 过滤掉无对应掩码的图像
            self.img_names = [f for f in self.img_names if os.path.exists(os.path.join(self.labels_dir, f))]
        # 1. 校验JSON文件是否存在
        else:
            if not os.path.exists(self.data_conf_json_path):
                raise FileNotFoundError(f"JSON文件不存在：{self.data_conf_json_path}")
            # 2. 读取并解析JSON文件
            try:
                with open(self.data_conf_json_path, 'r', encoding='utf-8') as f:
                    split_data_conf = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"JSON文件格式错误，无法解析：{self.data_conf_json_path}")
            # 3. 校验客户端名称是否存在
            cid = f"{self.client_id}"
            if cid not in split_data_conf:
                raise KeyError(
                    f"客户端{cid}不存在！已有的客户端：{list(split_data_conf.keys())}"
                )

            # 4. 提取train/test文件列表并校验字段
            client_data = split_data_conf[cid]
            if self.split not in client_data:
                raise ValueError(f"客户端{cid}的JSON数据缺少{self.split}字段")

            self.img_names = client_data[self.split]

            # 5. 校验字段类型（确保是列表）
            if not isinstance(self.img_names, list):
                raise ValueError(f"客户端{cid}的{self.split}必须是列表类型")

            # print(f"客户端{cid} {self.split} 文件数：{len(self.img_names)}")

    def __len__(self) -> int:
        """返回成对的样本总数"""
        return len(self.img_names)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        根据索引获取单个样本（语义分割场景）
        返回：(处理后的原始图片, 处理后的像素级标签掩码)
        """
        # 1. 获取当前样本的基础文件名（不带后缀）
        img_name = self.img_names[index]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name)

        # 2. 读取原始图片（遥感图片可能多通道，转为RGB/灰度按需调整）
        # 若你的遥感图是多光谱（>3通道），去掉.convert("RGB")，直接Image.open即可
        img = Image.open(img_path).convert("RGB")
        # 读取标签掩码（语义分割标签是单通道，转为L模式（8位灰度））
        label = Image.open(label_path).convert("L")

        # 3. 应用图片变换（如Resize、ToTensor、归一化等）
        if self.transform is not None:
            img = self.transform(img)

        # 4. 应用标签掩码变换（需和图片变换同步，比如同尺寸缩放）

        if self.target_transform is None:
            label = torch.from_numpy(np.array(label, dtype=np.int64))
        else:
            label = self.target_transform(label)
        label = label.squeeze().long()  # 掩码转为LongTensor（交叉熵损失要求）
        return img, label
