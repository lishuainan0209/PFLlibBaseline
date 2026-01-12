import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from flcore.clients.clientbase import Client
from utils.datasets import RemoteSensingSegDataset


class LoveDA2021RuralClient(Client):
    def __init__(self, args, id, train_samples=0, test_samples=0, **kwargs):
        super().__init__(args, id, train_samples, test_samples, train_slow=False, send_slow=False, **kwargs)
        self.data_conf_json = args.data_conf_json

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
        print(f"train client : {self.id},train_samples num: {self.train_samples} ")
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        # 模拟延迟
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

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

    def _compute_metrics_from_stats(self, stats, num_classes, total_samples):
        """
        基于累计的基础统计量计算最终指标
        """
        # 1. 总体准确率 OA
        oa = stats['correct_pixels'] / stats['total_pixels'] if stats['total_pixels'] > 0 else 0.0

        # 2. 类别级PA、Precision、Recall
        pa_per_class = []
        precision_per_class = []
        recall_per_class = []
        iou_per_class = []

        for cls in range(num_classes):
            tp = stats['tp'][cls]
            fp = stats['fp'][cls]
            fn = stats['fn'][cls]

            # PA: 该类正确数 / 该类总标签数
            cls_total = tp + fn
            if cls_total > 0:
                pa_per_class.append(tp / cls_total)
            else:
                pa_per_class.append(0.0)

            # Precision: TP/(TP+FP)
            if tp + fp > 0:
                precision_per_class.append(tp / (tp + fp))
            else:
                precision_per_class.append(0.0)

            # Recall: TP/(TP+FN)
            if tp + fn > 0:
                recall_per_class.append(tp / (tp + fn))
            else:
                recall_per_class.append(0.0)

            # IoU: TP/(TP+FP+FN)
            if tp + fp + fn > 0:
                iou_per_class.append(tp / (tp + fp + fn))
            else:
                iou_per_class.append(0.0)

        # 3. 平均指标
        mpa = np.mean(pa_per_class) if pa_per_class else 0.0
        miou = np.mean(iou_per_class) if iou_per_class else 0.0
        precision = np.mean(precision_per_class) if precision_per_class else 0.0
        recall = np.mean(recall_per_class) if recall_per_class else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 4. Kappa系数
        try:
            kappa = cohen_kappa_score(stats['pred_flatten'], stats['label_flatten'])
        except:
            kappa = 0.0

        # 5. 平均损失
        avg_loss = stats["total_loss"] / total_samples if total_samples > 0 else 0.0

        return {
            "avg_loss": avg_loss,
            "oa": oa,
            "mpa": mpa,
            "miou": miou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "kappa": kappa
        }

    # 评价指标
    def test_metrics(self):

        test_data_loader = self.load_test_data()
        print(f"test_metrics client: {self.id}, test_samples num: {self.test_samples} ")
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

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
            for images, labels in test_data_loader:  # tqdm(self.test_loader, desc="Test"):
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

        metrics = self._compute_metrics_from_stats(stats, self.num_classes, total_samples)

        # 返回：样本数、客户端最终指标,基础统计量
        return total_samples, metrics, stats

    def train_metrics(self):
        train_data_loader = self.load_train_data()
        print(f"train_metrics client: {self.id}, train_samples num: {self.train_samples} ")
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()
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
            for images, labels in train_data_loader:  # tqdm(self.train_loader, desc="Test"):
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

        # todo 加上时间消耗
        # self.train_time_cost['num_rounds'] += 1
        # self.train_time_cost['total_cost'] += time.time() - start_time
        # 返回：样本数、客户端最终指标,基础统计量
        return total_samples, metrics, stats

    # todo 训练完进行保存

    def _compute_segmentation_metrics(self, pred, label, num_classes):
        """
        计算单张图像的分割评价指标
        Args:
            pred: 预测的类别图 (H,W) tensor
            label: 真实标签图 (H,W) tensor
            num_classes: 类别总数
        Returns:
            单样本的各类指标：pa, mpa, miou, precision, recall, f1, oa, kappa
        """
        # 转换为numpy数组（方便计算）
        pred_np = pred.cpu().numpy().flatten()
        label_np = label.cpu().numpy().flatten()

        # 1. 总体准确率 (Overall Accuracy, OA)
        oa = (pred_np == label_np).sum() / len(label_np)

        # 2. 像素准确率 (Pixel Accuracy, PA)、平均像素准确率 (mPA)
        pa_per_class = []
        # 3. IoU 与 mIoU（复用原有逻辑）
        iou_per_class = []
        # 4. 精确率 (Precision)、召回率 (Recall)、F1-Score
        precision_per_class = []
        recall_per_class = []

        for cls in range(num_classes):
            # 该类的预测/标签掩码
            pred_cls = (pred_np == cls)
            label_cls = (label_np == cls)

            # PA：该类正确预测的像素数 / 该类总像素数
            cls_total = label_cls.sum()
            if cls_total > 0:
                pa_per_class.append((pred_cls & label_cls).sum() / cls_total)

            # IoU
            intersection = (pred_cls & label_cls).sum()
            union = (pred_cls | label_cls).sum()
            if union > 0:
                iou_per_class.append(intersection / union)

            # Precision (精确率)：TP/(TP+FP)
            pred_cls_total = pred_cls.sum()
            if pred_cls_total > 0:
                precision_per_class.append(intersection / pred_cls_total)
            else:
                precision_per_class.append(0.0)  # 无预测为该类时精确率为0

            # Recall (召回率)：TP/(TP+FN)
            if cls_total > 0:
                recall_per_class.append(intersection / cls_total)
            else:
                recall_per_class.append(0.0)  # 无该类标签时召回率为0

        # 计算各类指标的均值
        mpa = np.mean(pa_per_class) if pa_per_class else 0.0
        miou = np.mean(iou_per_class) if iou_per_class else 0.0
        precision = np.mean(precision_per_class) if precision_per_class else 0.0
        recall = np.mean(recall_per_class) if recall_per_class else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 5. Kappa系数（可选，衡量分类一致性）
        try:
            kappa = cohen_kappa_score(pred_np, label_np)
        except:
            kappa = 0.0  # 极端情况（如只有一个类别）避免报错

        return pa_per_class, mpa, miou, precision, recall, f1, oa, kappa
