from utils.datasets import RemoteSensingSegDataset
import  time
start_time_str = time.strftime("%Y%m%d-%H%M%S")
import torch
import numpy as np
from collections import defaultdict
# todo 可以根据统计程序, 来写划分的程序
# 1. 初始化你的数据集（保持和你原代码一致的参数）
for i in ["test", "train"]:
    dataset = RemoteSensingSegDataset(
        dataset_name=r"2021LoveDA_Urban",
        split=i,  # 可替换为 train/test 统计对应划分
        transform=None,
        target_transform=None,
        img_suffix=".png",
        data_conf_json="2021LoveDA_Urban.json",
        client_id=None,
    )

    # 2. 初始化统计容器（记录每个类的像素数）
    class_pixel_count = defaultdict(int)  # 键：类别ID（0-7），值：该类像素总数
    total_pixels = 0  # 所有像素总数

    # 3. 遍历数据集，统计每个样本的标签像素
    print(f"开始统计 {len(dataset)} 个样本的类别像素占比...")
    for idx in range(len(dataset)):
        # 获取单个样本的图片和标签（labels是HxW的张量/数组）
        img, labels = dataset[idx]

        # 处理标签格式：转为numpy数组（兼容张量/PIL Image），并展平为一维
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy().flatten()  # 张量转CPU→numpy→展平
        else:
            labels_np = np.array(labels).flatten()  # PIL Image直接转numpy→展平

        # 统计当前样本的各类别像素数
        unique_classes, counts = np.unique(labels_np, return_counts=True)
        for cls, cnt in zip(unique_classes, counts):
            cls = int(cls)  # 确保类别ID是整数
            class_pixel_count[cls] += cnt
        total_pixels += len(labels_np)

        # 可选：打印进度（每100个样本提示一次）
        # if (idx + 1) % 100 == 0:
        #     print(f"已统计 {idx + 1}/{len(dataset)} 个样本")

    # 4. 计算各类别占比（按0-7类排序，确保无遗漏）
    num_classes = 8  # LoveDA农村数据集0-7共8类
    class_ratio = {}
    print("\n===== 类别像素占比统计结果 =====")
    for cls in range(num_classes):
        cnt = class_pixel_count.get(cls, 0)
        ratio = cnt / total_pixels if total_pixels > 0 else 0.0
        class_ratio[cls] = ratio
        print(f"类别 {cls}：像素数 = {cnt:,} | 占比 = {ratio:.4f} ({ratio * 100:.2f}%)")

    # 5. 验证总占比（确保所有类别占比之和为1）
    total_ratio = sum(class_ratio.values())
    print(f"\n总像素数：{total_pixels:,} | 所有类别占比之和：{total_ratio:.4f}")