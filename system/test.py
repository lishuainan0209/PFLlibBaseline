from utils.datasets import RemoteSensingSegDataset

dataset = RemoteSensingSegDataset(root=r"D:\PFLlibBaseline\dataset\2021LoveDA_Rural",  # 数据集根路径（如"你的遥感数据集路径"）
                                  split="train",  # 选择数据集划分：train/val/test
                                  transform=None,  # 图片变换（Resize、ToTensor等）
                                  target_transform=None,  # 标签掩码变换
                                  img_suffix=".png",  # 图片后缀（适配遥感常见格式：png/tif/jpg）
                                  data_conf_json="2021LoveDA_Rural.json",  # 划分的数据集配置文件
                                  client_id=1,  # 节点名称)
                                  )
