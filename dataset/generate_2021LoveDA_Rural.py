# 将文件夹中图片按照联邦客户端数量划分成数据集,图片名作为标识
import json
import os
import random
from typing import List, Dict


# todo 固定随机种子
# 支持的图片/标签文件后缀（LoveDA数据集格式）


def load_split_files(data_dir: str, split: str) -> List[str]:
    """
    读取指定划分（train/test）下的图片文件名（确保对应label存在）
    Args:
        data_dir: 数据集根目录（如D:\mybaseline\datasets\2021LoveDA_Urban）
        split: 划分类型（"train"或"test"）
    Returns:
        排序后的图片文件名列表（已过滤无对应label的文件）
    """
    # 拼接图片和标签目录路径
    img_dir = os.path.join(data_dir, split, "images")
    label_dir = os.path.join(data_dir, split, "labels")

    # 校验目录是否存在
    for dir_path in [img_dir, label_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在：{dir_path}")

    # 筛选图片文件，并确保对应label存在
    img_filenames = []
    for filename in os.listdir(img_dir):
        if filename.startswith('.'):  # 跳过隐藏文件
            continue
        if filename.lower().endswith('.png'):
            # 检查对应的label是否存在
            label_path = os.path.join(label_dir, filename)
            if os.path.exists(label_path):
                img_filenames.append(filename)

    # 排序（保证划分结果可复现）
    img_filenames.sort()

    if not img_filenames:
        raise ValueError(f"{split}集下未找到有效图片文件（需同时存在对应label）")

    print(f"成功读取 {split} 集：{len(img_filenames)} 个文件")
    return img_filenames


def split_files_equally(filenames: List[str], client_num: int) -> List[List[str]]:
    """
    将文件列表按客户端数量均等划分（余数分配给前N个客户端）
    Returns:
        划分后的列表，每个元素是对应客户端的文件列表
    """
    if client_num <= 0:
        raise ValueError("客户端数量必须大于0")

    total = len(filenames)
    base_num = total // client_num
    remainder = total % client_num

    splits = []
    start = 0
    random.shuffle(filenames)
    for i in range(client_num):
        end = start + base_num + (1 if i < remainder else 0)
        splits.append(filenames[start:end])
        start = end
    return splits


def save_client_split_to_json(client_data: Dict, DATASET_ROOT, JSON_NAME):
    """将客户端划分结果保存为JSON"""
    # 创建保存目录（若不存在）
    save_path = os.path.join(DATASET_ROOT, JSON_NAME)
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存JSON（保证可读性）
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(client_data, f, ensure_ascii=False, indent=4)

    print(f"划分结果已保存至：{save_path}")


# 直接右键运行即可, 无需配置,pycharm会自动以该py所在文件夹为工作目录
if __name__ == "__main__":
    # -------------------------- 配置参数（按需修改） --------------------------
    # 2021LoveDA_Rural   2021LoveDA_Urban
    DATASET_ROOT = r"2021LoveDA_Rural/"  # 数据集根目录
    # todo 加上server验证集以及划分
    CLIENT_NUM = 16  # 联邦学习客户端数量
    JSON_NAME = r"2021LoveDA_Rural.json"  # 结果保存路径

    # 1. 读取train和test集的文件列表
    train_files = load_split_files(DATASET_ROOT, split="train")
    test_files = load_split_files(DATASET_ROOT, split="test")

    # 2. 分别对train和test集做均等划分
    # todo 不均等划分如何做?
    train_splits = split_files_equally(train_files, CLIENT_NUM)
    test_splits = split_files_equally(test_files, CLIENT_NUM)

    # 3. 构造每个客户端的train/test文件映射
    client_split_data = {}
    client_split_data["info"] = {
        "desc": "随意均分",
        "num_clients": CLIENT_NUM,
    }
    for client_idx in range(CLIENT_NUM):
        client_name = f"{client_idx}"  # 转成json后, 会自动变成str,不如直接变
        client_split_data[client_name] = {
            "other": "可以放一些其他数据, 如是否慢速训练等",
            "train": train_splits[client_idx],
            "test": test_splits[client_idx]
        }
        # 打印每个客户端的分配情况
        print(f"{client_name} - train: {len(train_splits[client_idx])}个文件 | test: {len(test_splits[client_idx])}个文件")

    # 4. 保存为JSON
    save_client_split_to_json(client_split_data, DATASET_ROOT, JSON_NAME)
