import time

import numpy as np
import torch
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader
from utils.datasets import RemoteSensingSegDataset


class LoveDA2021RuralClient(Client):
    def __init__(self, args, id, train_samples=0, test_samples=0, **kwargs):
        super().__init__(args, id, train_samples, test_samples, train_slow=False, send_slow=False, **kwargs)
        self.data_conf_json=args.data_conf_json
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
        print(f"{self.id} train")
        trainloader = self.load_train_data()
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

    # 评价指标
    def test_metrics(self):
        print(f"{self.id} test_metrics")
        test_data_loader = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_IoU = 0.0
        with torch.no_grad():
            for images, labels in test_data_loader:  # tqdm(self.test_loader, desc="Test"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                total_loss += loss.item() * images.size(0)
                # todo 多个评价指标
                # 计算mIoU
                preds = torch.argmax(outputs, dim=1)  # 预测类别(取概率最大的类)
                for pred, label in zip(preds, labels):
                    # 计算每个类的IoU
                    IoU_per_class = []
                    for cls in range(self.num_classes):
                        pred_cls = (pred == cls)
                        label_cls = (label == cls)
                        intersection = (pred_cls & label_cls).sum().item()
                        union = (pred_cls | label_cls).sum().item()
                        if union > 0:
                            IoU_per_class.append(intersection / union)
                    # 平均IoU(mIoU)
                    if IoU_per_class:
                        total_IoU += np.mean(IoU_per_class)

        # avg_loss = total_loss / len(self.test_loader.dataset)
        # avg_mIoU = total_IoU / len(self.test_loader.dataset)
        return len(test_data_loader.dataset), total_loss, total_IoU

    # todo 就不能在训练后就保存之吗?考虑到训练时模型不稳定?这里是测试?
    def train_metrics(self):
        print(f"{self.id} train_metrics")
        train_data_loader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_IoU = 0.0
        with torch.no_grad():
            for images, labels in train_data_loader:  # tqdm(self.test_loader, desc="Test"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                total_loss += loss.item() * images.size(0)
                # todo 多个评价指标
                # 计算mIoU
                preds = torch.argmax(outputs, dim=1)  # 预测类别(取概率最大的类)
                for pred, label in zip(preds, labels):
                    # 计算每个类的IoU
                    IoU_per_class = []
                    for cls in range(self.num_classes):
                        pred_cls = (pred == cls)
                        label_cls = (label == cls)
                        intersection = (pred_cls & label_cls).sum().item()
                        union = (pred_cls | label_cls).sum().item()
                        if union > 0:
                            IoU_per_class.append(intersection / union)
                    # 平均IoU(mIoU)
                    if IoU_per_class:
                        total_IoU += np.mean(IoU_per_class)

        # avg_loss = total_loss / len(self.test_loader.dataset)
        # avg_mIoU = total_IoU / len(self.test_loader.dataset)
        return len(train_data_loader.dataset), total_loss, total_IoU
    # todo 训练完进行保存
