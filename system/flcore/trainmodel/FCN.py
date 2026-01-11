import torch
import torch.nn as nn


# 完整的FCN8s网络（无预训练、纯手动构建）
class FCN8s(nn.Module):
    def __init__(self, num_classes=8):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        # todo 每一层参数初始化的赋值问题,正态分布赋值
        # -------------------------- 1. 手动构建VGG16的特征提取部分（无预训练） --------------------------
        self.pool1 = nn.Sequential(
            # 第1组：Conv+Conv+Pool (pool1)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样2倍
        )
        self.pool2 = nn.Sequential(
            # 第2组：Conv+Conv+Pool (pool2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样2倍（累计4倍）
        )
        self.pool3 = nn.Sequential(
            # 第3组：Conv+Conv+Conv+Pool (pool3) —— 跳跃连接1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样2倍（累计8倍）
        )
        self.pool4 = nn.Sequential(
            # 第4组：Conv+Conv+Conv+Pool (pool4) —— 跳跃连接2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样2倍（累计16倍）
        )
        self.pool5 = nn.Sequential(
            # 第5组：Conv+Conv+Conv+Pool (pool5)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样2倍（累计32倍）
        )

        # -------------------------- 2. FCN分类器 --------------------------
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # -------------------------- 3. 跳跃连接1x1卷积 --------------------------
        self.pool3_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        self.pool4_conv = nn.Conv2d(512, num_classes, kernel_size=1)

        # -------------------------- 4. 上采样层（优化padding + 初始化权重） --------------------------
        self.upsample2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False)
        # 优化：添加padding=1，适配非8倍数的输入尺寸
        self.upsample8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8, bias=False)

        # 关键：初始化转置卷积权重（提升训练收敛速度）
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_h, input_w = x.size()[2], x.size()[3]

        # 前向传播：逐组提取特征
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        pool3 = x  # 捕获pool3特征（8倍下采样）
        x = self.pool4(x)
        pool4 = x  # 捕获pool4特征（16倍下采样）
        x = self.pool5(x)
        x = self.classifier(x)  # pool5特征转类别通道

        # 跳跃连接1：pool5→pool4
        pool4 = self.pool4_conv(pool4)
        x = self.upsample2x(x)
        x = x[:, :, :pool4.size(2), :pool4.size(3)]  # 尺寸对齐
        x += pool4

        # 跳跃连接2：pool4→pool3
        pool3 = self.pool3_conv(pool3)
        x = self.upsample2x(x)
        x = x[:, :, :pool3.size(2), :pool3.size(3)]  # 尺寸对齐
        x += pool3

        # 最终上采样到输入尺寸
        x = self.upsample8x(x)
        x = x[:, :, :input_h, :input_w]  # 最终尺寸对齐
        return x
    def save_model_parameters(self,checkpoint_path:str,*args,**kwargs):
        """根据文件进行模型保存"""
        torch.save({
            "epoch": kwargs["epoch"],
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": kwargs["optimizer"].state_dict(),
            "best_mIoU": kwargs["best_test_mIoU"],
        }, checkpoint_path)

    # todo 用于验证的main
    def load_model_parameters(self,checkpoint_path:str,*args,**kwargs):
        """根据路径加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 先加载到CPU，避免设备不匹配
        self.load_state_dict(checkpoint["model_state_dict"])
        # 5. （可选）预测/验证时，设置模型为评估模式
        self.eval()
