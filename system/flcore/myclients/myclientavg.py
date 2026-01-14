import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from flcore.clients.clientbase import Client
from utils.datasets import RemoteSensingSegDataset


class LoveDA2021RuralClient(Client):
    def __init__(self, args, id, train_samples=0, test_samples=0, **kwargs):
        super().__init__(args, id, train_samples, test_samples, train_slow=False, send_slow=False, **kwargs)
        self.data_conf_json = args.data_conf_json
    #     todo train_slow和 send_slow要基于模型大小,宽带(百兆,千兆,参数设置), ping 某个网站的返回毫秒数(或者参数设置RTT,高斯随机, 高于某个值就代表离线), 建立一个数学模型,模拟网络延迟
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = RemoteSensingSegDataset(dataset_name=self.dataset,  # 数据集根路径（如"你的遥感数据集路径"）
                                             split="train",  # 选择数据集划分：train/val/test
                                             transform=None,  # 图片变换（Resize、ToTensor等）
                                             target_transform=None,  # 标签掩码变换
                                             img_suffix=".png",  # 图片后缀（适配遥感常见格式：png/tif/jpg）
                                             data_conf_json=self.data_conf_json,  # 划分的数据集配置文件
                                             client_id=self.id,  # 节点名称
                                             )
        # read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        self.train_samples = len(train_data)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = RemoteSensingSegDataset(dataset_name=self.dataset,  # 数据集根路径（如"你的遥感数据集路径"）
                                            split="test",  # 选择数据集划分：train/val/test
                                            transform=None,  # 图片变换（Resize、ToTensor等）
                                            target_transform=None,  # 标签掩码变换
                                            img_suffix=".png",  # 图片后缀（适配遥感常见格式：png/tif/jpg）
                                            data_conf_json=self.data_conf_json,  # 划分的数据集配置文件
                                            client_id=self.id,  # 节点名称
                                            )
        self.test_samples = len(test_data)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def train(self):
        trainloader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        # 模拟延迟
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for images, labels in tqdm(trainloader, desc=f"train client {self.id}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(images)
                loss = self.loss(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 训练完成后，快速迁移到CPU
        self.model.cpu()
        print(f"train client : {self.id:03d},train_samples num: {self.train_samples:04d}, cost time: {time.time() - start_time:.2f} s  ")
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _accumulate_base_stats(self, pred, label, num_classes, stats):
        """
        累计单张图像的基础统计量（核心修改）
        Args:
            pred: 预测图 (H,W) tensor
            label: 标签图 (H,W) tensor
            num_classes: 类别数
            stats: 统计量字典（原地更新）
        """
        pred_np = pred.cpu().numpy().flatten()
        label_np = label.cpu().numpy().flatten()

        # 累计总像素数和正确像素数（用于OA）
        stats['total_pixels'] += len(label_np)
        stats['correct_pixels'] += (pred_np == label_np).sum()

        # 累计预测/标签数组（用于kappa）
        stats['pred_flatten'].extend(pred_np.tolist())
        stats['label_flatten'].extend(label_np.tolist())

        # 累计每个类别的TP/FP/FN
        for cls in range(num_classes):
            pred_cls = (pred_np == cls)
            label_cls = (label_np == cls)

            # TP: 预测为cls且标签为cls
            stats['tp'][cls] += (pred_cls & label_cls).sum()
            # FP: 预测为cls但标签不是cls
            stats['fp'][cls] += (pred_cls & ~label_cls).sum()
            # FN: 标签为cls但预测不是cls
            stats['fn'][cls] += (~pred_cls & label_cls).sum()

    def cohen_kappa_score_gpu(self,y_true, y_pred, num_classes):
        """
        纯 GPU 实现的 Cohen's Kappa 系数计算（无 CPU 拷贝，并行计算）
        :param y_true: GPU 张量，形状 (N,)，标签值（整数）
        :param y_pred: GPU 张量，形状 (N,)，预测值（整数）
        :param num_classes: 总类别数（int）
        :return: Kappa 系数（GPU 张量，标量）
        """
        # 确保输入是 GPU 上的长整型张量
        y_true = y_true.long().to(self.device)
        y_pred = y_pred.long().to(self.device)

        # 1. 计算混淆矩阵（GPU 并行）
        conf_mat = torch.bincount(
            num_classes * y_true + y_pred,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        # 2. 计算观测一致率 Po（对角线总和 / 总样本数）
        total_samples = conf_mat.sum()
        if total_samples == 0:
            return torch.tensor(0.0, device=self.device)
        po = torch.diag(conf_mat).sum() / total_samples

        # 3. 计算期望一致率 Pe（行和 × 列和 的总和 / 总样本数²）
        row_sum = conf_mat.sum(dim=1)  # 每行和（真实类别数）
        col_sum = conf_mat.sum(dim=0)  # 每列和（预测类别数）
        pe = (row_sum * col_sum).sum() / (total_samples ** 2)

        # 4. 计算 Kappa 系数（避免除以 0）
        kappa = torch.where(
            (1 - pe) > 1e-6,
            (po - pe) / (1 - pe),
            torch.tensor(0.0, device=self.device)
        )
        return kappa

    def _compute_metrics_from_stats(self, stats, num_classes, total_samples):
        # 1. 计时：数据拷贝+张量转换
        t0 = time.time()
        # 【原张量转换代码】
        tp = torch.tensor(stats['tp'], dtype=torch.float32, device=self.device)
        fp = torch.tensor(stats['fp'], dtype=torch.float32, device=self.device)
        fn = torch.tensor(stats['fn'], dtype=torch.float32, device=self.device)
        correct_pixels = torch.tensor(stats['correct_pixels'], dtype=torch.float32, device=self.device)
        total_pixels = torch.tensor(stats['total_pixels'], dtype=torch.float32, device=self.device)
        total_loss = torch.tensor(stats['total_loss'], dtype=torch.float32, device=self.device)
        total_samples = torch.tensor(total_samples, dtype=torch.float32, device=self.device)
        t1 = time.time()
        # print(f"数据拷贝+张量转换耗时：{t1 - t0:.4f}s")

        # 2. 计时：GPU 指标计算
        oa = torch.where(total_pixels > 0, correct_pixels / total_pixels, torch.tensor(0.0, device=self.device))
        cls_total = tp + fn
        pa_per_class = torch.where(cls_total > 0, tp / cls_total, torch.tensor(0.0, device=self.device))
        precision_denominator = tp + fp
        precision_per_class = torch.where(precision_denominator > 0, tp / precision_denominator, torch.tensor(0.0, device=self.device))
        recall_denominator = tp + fn
        recall_per_class = torch.where(recall_denominator > 0, tp / recall_denominator, torch.tensor(0.0, device=self.device))
        iou_denominator = tp + fp + fn
        iou_per_class = torch.where(iou_denominator > 0, tp / iou_denominator, torch.tensor(0.0, device=self.device))
        mpa = torch.mean(pa_per_class) if pa_per_class.numel() > 0 else torch.tensor(0.0, device=self.device)
        miou = torch.mean(iou_per_class) if iou_per_class.numel() > 0 else torch.tensor(0.0, device=self.device)
        precision = torch.mean(precision_per_class) if precision_per_class.numel() > 0 else torch.tensor(0.0, device=self.device)
        recall = torch.mean(recall_per_class) if recall_per_class.numel() > 0 else torch.tensor(0.0, device=self.device)
        f1_denominator = precision + recall
        f1 = torch.where(f1_denominator > 0, 2 * (precision * recall) / f1_denominator, torch.tensor(0.0, device=self.device))
        t2 = time.time()
        # print(f"GPU 指标计算耗时：{t2 - t1:.4f}s")

        # 3. 计时：Kappa 系数（核心瓶颈）
        # todo 太费时间了
        # kappa=0
        # ===================== 优化后的 Kappa 系数计算（GPU 版） =====================
        # try:
        #     # 1. 将 pred/label 转为 GPU 张量（仅 1 次拷贝，耗时 < 0.01s）
        #     pred_flatten = torch.tensor(stats['pred_flatten'], device=self.device)
        #     label_flatten = torch.tensor(stats['label_flatten'], device=self.device)
        #     # 2. 调用 GPU 版 Kappa 函数（核心优化）
        #     kappa = self.cohen_kappa_score_gpu(label_flatten, pred_flatten, num_classes)
        # except Exception as e:
        #     print(f"Kappa 计算出错：{e}")
        #     kappa = torch.tensor(0.0, device=self.device)
        t3 = time.time()
        # print(f"Kappa 系数计算耗时：{t3 - t2:.4f}s")
        # 4. 平均损失 + 结果转回 CPU
        avg_loss = torch.where(total_samples > 0, total_loss / total_samples, torch.tensor(0.0, device=self.device))
        t4 = time.time()
        print("precision_per_class",precision_per_class.tolist())
        # todo 这些指标能大写的就大写
        res = {
            "avg_loss": avg_loss.cpu().item(),
            "precision": precision.cpu().item(),
            "oa": oa.cpu().item(),
            "mpa": mpa.cpu().item(),
            "miou": miou.cpu().item(),

            "recall": recall.cpu().item(),
            "f1": f1.cpu().item(),
            # "precision_per_class":precision_per_class.cpu().tolist(),
            # "kappa": kappa.cpu().item(),
            "kappa": 0
        }
        # print(f"\n client {self.id} 计算评价指标耗时：{t4 - t0:.4f} s\n")
        return res
    # 评价指标
    def test_metrics(self):

        test_data_loader = self.load_test_data()

        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()
        start_time = time.time()
        # 初始化基础统计量（仅累计，不计算指标）
        stats = {
            "total_pixels": 0,
            "correct_pixels": 0,
            "total_loss": 0,
            "tp": np.zeros(self.num_classes, dtype=np.int64),
            "fp": np.zeros(self.num_classes, dtype=np.int64),
            "fn": np.zeros(self.num_classes, dtype=np.int64),
            "pred_flatten": [],
            "label_flatten": []
        }
        # 初始化所有指标累加器
        total_samples = 0  # 客户端本地样本总数

        with torch.no_grad():
            for images, labels in tqdm(test_data_loader, desc=f"test_metrics client {self.id}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                stats["total_loss"] += loss.item() * images.size(0)
                total_samples += images.size(0)

                preds = torch.argmax(outputs, dim=1)  # 预测类别(取概率最大的类)
                # 遍历每个样本，累计基础统计量
                for pred, label in zip(preds, labels):
                    self._accumulate_base_stats(pred, label, self.num_classes, stats)
        # 计算完成后，快速迁移到CPU
        self.model.cpu()
        metrics = self._compute_metrics_from_stats(stats, self.num_classes, total_samples)
        # todo 计算出来的metrics保存起来, 每一轮和上一次进行比较,显示出来增加或者减少
        print(f"test_metrics client: {self.id:03d}, test_samples num: {self.test_samples:04d}, cost_time: {time.time() - start_time:.2f} s \n{metrics} \n")
        # 返回：样本数、客户端最终指标,基础统计量, stats会把内存吃完, 导致程序停止, 先不返回
        # return total_samples, metrics, stats
        return total_samples, metrics, {}

    def train_metrics(self):
        train_data_loader = self.load_train_data()

        # self.model = self.load_model('model')
        # print(self.device)
        self.model.to(self.device)
        self.model.eval()
        start_time = time.time()
        # 初始化基础统计量
        stats = {
            "total_pixels": 0,
            "correct_pixels": 0,
            "total_loss": 0,
            "tp": np.zeros(self.num_classes, dtype=np.int64),
            "fp": np.zeros(self.num_classes, dtype=np.int64),
            "fn": np.zeros(self.num_classes, dtype=np.int64),
            "pred_flatten": [],
            "label_flatten": []
        }
        # 初始化所有指标累加器
        total_samples = 0  # 客户端本地样本总数

        with torch.no_grad():
            for images, labels in tqdm(train_data_loader, desc=f"train_metrics client {self.id}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                stats["total_loss"] += loss.item() * images.size(0)
                total_samples += images.size(0)

                preds = torch.argmax(outputs, dim=1)  # 预测类别(取概率最大的类)
                # 遍历每个样本计算指标
                for pred, label in zip(preds, labels):
                    self._accumulate_base_stats(pred, label, self.num_classes, stats)

        metrics = self._compute_metrics_from_stats(stats, self.num_classes, total_samples)
        # 训练完成后，快速迁移到CPU
        self.model.cpu()
        # todo 计算出来的metrics保存起来, 每一轮和上一次进行比较,显示出来增加或者减少
        print(f"train_metrics client: {self.id:03d}, train_samples num: {self.train_samples:04d},cost_time: {time.time() - start_time:.2f} s \n{metrics} \n")
        # 返回：样本数、客户端最终指标,基础统计量,stats会把内存吃完, 导致程序停止, 先不返回
        # return total_samples, metrics, stats
        return total_samples, metrics, {}
