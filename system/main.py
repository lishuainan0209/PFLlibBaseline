#!/usr/bin/env python
import argparse
import copy
import logging
import os
import time
import warnings

import numpy as np
import torchvision

from flcore.servers.serverala import FedALA
from flcore.servers.serveramp import FedAMP
from flcore.servers.serverapfl import APFL
from flcore.servers.serverapple import APPLE
from flcore.servers.serveras import FedAS
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverbn import FedBN
from flcore.servers.servercac import FedCAC
from flcore.servers.servercp import FedCP
from flcore.servers.servercross import FedCross
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverdbe import FedDBE
from flcore.servers.serverditto import Ditto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.serverfd import FD
from flcore.servers.serverfml import FML
from flcore.servers.serverfomo import FedFomo
from flcore.servers.servergc import FedGC
from flcore.servers.servergen import FedGen
from flcore.servers.servergh import FedGH
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverkd import FedKD
from flcore.servers.serverlc import FedLC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverlocal import Local
from flcore.servers.servermoon import MOON
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverntd import FedNTD
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverpcl import FedPCL
from flcore.servers.serverper import FedPer
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverproto import FedProto
from flcore.servers.serverprox import FedProx
from flcore.servers.serverrep import FedRep
from flcore.servers.serverrod import FedROD
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.myservers.myserveravg import LoveDA2021RuralFedAvg
# 模型
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.FCN import *
# 工具
from utils.mem_utils import MemReporter
from utils.result_utils import average_data
from utils.tool import set_seed

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def run(args):
    time_list = []
    mem_reporter = MemReporter()
    model_str = args.model
    # todo 每次循环都会产生新的模型以及服务端, 是否有必要修改该逻辑
    for i in range(args.prev, args.times):
        print("", "@" * 40)
        print(f"Running time: {i}th ", "=" * 10)
        print("Creating server and clients ...")
        start = time.time()

        # 根据 model_str和dataset获取对应的实际模型
        # Generate args.model
        if model_str == "FCN8s":
            args.model = FCN8s(num_classes=args.num_classes)
        elif model_str == "MLR":  # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(
                input_size=args.vocab_size,
                hidden_size=args.feature_dim,
                output_size=args.num_classes,
                num_layers=1,
                embedding_dropout=0,
                lstm_dropout=0,
                attention_dropout=0,
                embedding_length=args.feature_dim,
            ).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(
                args.device
            )

        elif model_str == "TextCNN":
            args.model = TextCNN(
                hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, num_classes=args.num_classes
            ).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(
                ntoken=args.vocab_size,
                d_model=args.feature_dim,
                nhead=8,
                nlayers=2,
                num_classes=args.num_classes,
                max_len=args.max_len,
            ).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == "HAR":
                args.model = HARCNN(
                    9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)
                ).to(args.device)
            elif args.dataset == "PAMAP2":
                args.model = HARCNN(
                    9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)
                ).to(args.device)

        else:
            raise NotImplementedError

        print("", args.model)

        # 设置聚合策略
        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            # 在联邦学习中，部分客户端的模型需要去掉某层（如全连接层），但又不想重写整个模型类，可用 nn.Identity() 占位
            args.model.fc = nn.Identity()  # nn.Identity() 几乎不会单独使用，核心价值是在不修改模型整体结构的前提下，替换 / 跳过某一层
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == "LoveDA2021RuralFedAvg":
            server = LoveDA2021RuralFedAvg(args, i)
        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == "FedCAC":
            server = FedCAC(args, i)

        elif args.algorithm == "PFL-DA":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)

        elif args.algorithm == "FedLC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == "FedAS":

            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedCross":
            server = FedCross(args, i)

        else:
            raise NotImplementedError
        # 进行训练
        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    mem_reporter.report()


def get_args():
    """
    nohup python main.py --seed=0 --device_id=3 \
    --num_clients=10 \
    --global_rounds=4 --local_epochs=2 --batch_size=4 \
        > $(date +%Y%m%d_%H%M%S).log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="PFLlib 联邦学习实验参数配置")
    # action = "store_true"
    # 开启该参数（设为    True）：命令行中只写参数名（无需加 = True / 1    等值）；
    # 关闭该参数（设为    False）：命令行中不写该参数即可（默认值就是    False）。
    # ===================== 1. 实验基础配置（全局通用） =====================
    parser.add_argument("-go", "--goal", type=str, default="test", help="实验目标：如 'test'（测试）、'train'（训练）、'privacy_analysis'（隐私分析）等", )
    parser.add_argument("-sd", "--seed", type=int, default=0, help="随机数种子：固定种子保证实验可复现")
    parser.add_argument("-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="训练设备：cpu/cuda（优先用GPU）")
    parser.add_argument("-did", "--device_id", type=str, default="3", help="GPU卡号（多GPU场景）：如 '0'（单卡）、'0,1'（多卡），仅device=cuda时生效")

    # ===================== 2. 数据集相关 =====================
    parser.add_argument("-data", "--dataset", type=str, default="2021LoveDA_Rural", help="数据集名称：如 MNIST/CIFAR10/Shakespeare/AG_News 等")
    parser.add_argument("-datajson", "--data_conf_json", type=str, default="2021LoveDA_Rural.json", help="数据集配置")
    parser.add_argument("-ncl", "--num_classes", type=int, default=8, help="数据集类别数：MNIST=10，CIFAR10=10，Shakespeare=80（文本）等")

    # ===================== 3. 客户端调度相关 =====================
    parser.add_argument("-nc", "--num_clients", type=int, default=2, help="联邦系统总客户端数量：静态联邦场景的基础规模")
    parser.add_argument("-nnc", "--num_new_clients", type=int, default=0, help="每轮新增客户端数：>0 时为动态联邦（模拟新设备加入）")
    parser.add_argument("-jr", "--join_ratio", type=float, default=1.0, help="每轮参与训练的客户端比例：0<jr≤1.0，1.0=全量参与，0.5=50%参与")
    parser.add_argument("-rjr", "--random_join_ratio", action="store_true", default=False,
                        help="是否随机参与比例：开启后每轮参与数在 0~join_ratio×num_clients 之间（模拟客户端随机在线）", )

    # ===================== 4. 联邦训练核心参数 =====================
    parser.add_argument("-pv", "--prev", type=int, default=0, help="断点续跑：之前中断的实验轮数，设置后从该轮继续训练（避免重复跑）")
    parser.add_argument("-t", "--times", type=int, default=1, help="实验重复次数：多次运行取平均，提升结果可靠性")
    parser.add_argument("-gr", "--global_rounds", type=int, default=2, help="全局聚合轮数：一次实验的总联邦通信轮数")
    parser.add_argument("-ls", "--local_epochs", type=int, default=1, help="客户端本地训练轮数：每轮联邦通信中，单个客户端本地训练的epoch数")
    parser.add_argument("-lbs", "--batch_size", type=int, default=4, help="客户端训练批次大小：每个batch的样本数量")

    parser.add_argument("-bnpc", "--batch_num_per_client", type=int, default=2, help="客户端每轮训练批次数：模拟算力受限，仅训练指定批次（而非全量数据）", )
    # ===================== 5. 模型/算法配置 =====================
    parser.add_argument("-m", "--model", type=str, default="FCN8s", help="模型架构：如 CNN（图像）、LSTM/Transformer（文本）等")
    parser.add_argument("-algo", "--algorithm", type=str, default="LoveDA2021RuralFedAvg", help="联邦聚合策略：FedAvg/FedProx/FedBN/FedPer 等")
    parser.add_argument("-lr", "--local_learning_rate", type=float, default=0.005, help="客户端本地基础学习率：控制参数更新步长")
    parser.add_argument("-ld", "--learning_rate_decay", action="store_true", default=False, help="是否开启本地学习率衰减：开启后每轮按gamma降低学习率")
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99, help="学习率衰减系数：仅ld=True时生效，0.99=每轮衰减1%，0.9=衰减10%", )

    # ===================== 6. 评估/终止/结果存储 =====================
    parser.add_argument("-ab", "--auto_break", action="store_true", default=False, help="是否开启训练自动终止：Top N客户端性能稳定时停止，避免无效轮次")
    parser.add_argument("-tc", "--top_cnt", type=int, default=100, help="auto_break阈值：选取前N个客户端的性能判断是否终止（总客户端<N时取全部）")
    parser.add_argument("-eg", "--eval_gap", type=int, default=1, help="评估间隔轮数：每N轮联邦训练后，用测试集评估全局模型性能")
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items", help="实验结果保存文件夹：存储模型参数、评估日志、DLG报告、训练记录等")

    # ===================== 7. 隐私分析（DLG） =====================
    parser.add_argument("-dlg", "--dlg_eval", action="store_true", default=False, help="是否开启DLG评估：通过梯度反推客户端数据，分析隐私泄露风险")
    parser.add_argument("-dlgg", "--dlg_gap", type=int, default=100, help="DLG评估间隔轮数：仅dlg_eval=True时生效，降低高频分析的计算开销")

    # ===================== 8. 动态客户端适配 =====================
    parser.add_argument("-ften", "--fine_tuning_epoch_new", type=int, default=0, help="新增客户端微调轮数：仅nnc>0时生效，先微调再参与联邦，避免模型震荡",
                        )

    # ===================== 9. 文本任务专属（非文本任务可忽略） =====================
    parser.add_argument("-fd", "--feature_dim", type=int, default=512, help="特征提取层输出维度：Base+Head拆分模型中，Base的输出维度（需匹配Head输入）")
    parser.add_argument("-vs", "--vocab_size", type=int, default=80, help="文本词汇表大小：Shakespeare=80，AG_News/SogouNews=32000")
    parser.add_argument("-ml", "--max_len", type=int, default=200, help="文本序列最大长度：超过截断，不足补0（统一输入维度）")

    # ===================== 10. 少样本场景适配 =====================
    parser.add_argument("-fs", "--few_shot", type=int, default=0, help="客户端少样本数：>0时每个客户端仅用N个样本训练（模拟数据稀缺场景）")

    # ===================== 11. 其他 =====================
    # practical
    parser.add_argument("-cdr", "--client_drop_rate", type=float, default=0.0, help="Rate for clients that train but drop out")
    parser.add_argument("-tsr", "--train_slow_rate", type=float, default=0.0, help="The rate for slow clients when training locally")
    parser.add_argument("-ssr", "--send_slow_rate", type=float, default=0.0, help="The rate for slow clients when sending global model")
    parser.add_argument("-ts", "--time_select", type=bool, default=False, help="Whether to group and select clients at each round according to time cost",
                        )
    parser.add_argument("-tth", "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument("-bt", "--beta", type=float, default=0.0)
    parser.add_argument("-lam", "--lamda", type=float, default=1.0, help="Regularization weight")
    parser.add_argument("-mu", "--mu", type=float, default=0.0)
    parser.add_argument("-K", "--K", type=int, default=5, help="Number of personalized training steps for pFedMe")
    parser.add_argument("-lrp", "--p_learning_rate", type=float, default=0.01, help="personalized learning rate to caculate theta aproximately using K steps", )
    # FedFomo
    parser.add_argument("-M", "--M", type=int, default=5, help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument("-itk", "--itk", type=int, default=4000, help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument("-alk", "--alphaK", type=float, default=1.0, help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument("-sg", "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument("-al", "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument("-pls", "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument("-tau", "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument("-fte", "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument("-dlr", "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument("-L", "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument("-nd", "--noise_dim", type=int, default=512)
    parser.add_argument("-glr", "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=512)
    parser.add_argument("-se", "--server_epochs", type=int, default=1000)
    parser.add_argument("-lf", "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument("-slr", "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument("-et", "--eta", type=float, default=1.0)
    parser.add_argument("-s", "--rand_percent", type=int, default=80)
    parser.add_argument("-p", "--layer_idx", type=int, default=2, help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument("-mlr", "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument("-Ts", "--T_start", type=float, default=0.95)
    parser.add_argument("-Te", "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument("-mo", "--momentum", type=float, default=0.1)
    parser.add_argument("-klw", "--kl_weight", type=float, default=0.0)

    # FedCross
    parser.add_argument("-fsb", "--first_stage_bound", type=int, default=0)
    parser.add_argument("-ca", "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument("-cmss", "--collaberative_model_select_strategy", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(12)
    total_start = time.time()
    args = get_args()
    # 设置expandable_segments:True，解决显存碎片化问题, 如果不设置, 会出现有10个G,但是显存都是1G不连续,无法分配到1G以上的空间
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 设置只看到一个gpu即可
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))
    print("=" * 50)

    run(args)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
