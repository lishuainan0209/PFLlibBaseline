import  json
import os
import time


import numpy as np

from flcore.myclients.myclientavg import LoveDA2021RuralClient
from flcore.servers.serverbase import Server


class LoveDA2021RuralFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(LoveDA2021RuralClient)
        print(f"\ntotal clients:{self.num_clients}, Join ratio:{self.join_ratio} ")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.per_round_metrics_results = dict()

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            client = clientObj(self.args, id=i, )
            self.clients.append(client)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            client = clientObj(self.args, id=i)
            self.clients.append(client)

    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            print(f"\nRound number: {i}")

            # 零. 首轮发送模型, 评估旧模型（基线）
            if i == 0 :
                self.send_models()
                print("\nEvaluate initial global model in client")
                self.evaluate()

            # 一. 选中的客户端进行训练
            print(f"\nStep 1: Train ")
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # 二.本地模型训练效果评估
            # 如果做个性化, 加入这步?; 做全局模型, 注释本步
            if  i % self.eval_gap == 0:
                print(f"\nStep 2: Evaluate Local model  in client,")
                self.evaluate()
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]


            # 三. 聚合模型,发送之
            print(f"\nStep 3: receive_models, aggregate_parameters, send_models, ")
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.send_models()

            # 四. 评价聚合后的新模型
            if  i % self.eval_gap == 0:
                print(f"\nStep 4: Evaluate  model in client (after aggregation),")
                self.evaluate()
            self.Budget.append(time.time() - s_t)
            print( f'time cost:{self.Budget[-1]:.4f}')
            print("#"*64)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break
        print("\n\n")
        for key in self.per_round_metrics_results:
            if "test_" in key:
                print(f"{key:<26}: min:{min(self.per_round_metrics_results[key]):.8f}   max:{max(self.per_round_metrics_results[key]):.8f}", )

        print(f"\nAverage time cost per round: {sum(self.Budget[1:]) / len(self.Budget[1:]):.4f} s")

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(LoveDA2021RuralClient)
        #     print(f"\nFine tuning round", "-" * 20)
        #     print("\nEvaluate new clients")
        #     self.evaluate()

    def save_results(self):
        jsonName = self.dataset + "_" + self.algorithm + "_" + self.goal + "_" + str(self.times)
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_path = os.path.join(result_path,f"{jsonName}.json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.per_round_metrics_results, f, ensure_ascii=False, indent=4)
        print("result File path: " + file_path)

    def evaluate(self, acc=None, loss=None):
        # 服务端本地评估
        # 进行评估,对本轮产出的模型,在训练集测试集上都进行评估,
        # total_stats中的每个元素
        # stats = {
        #     "total_pixels": 0,
        #     "correct_pixels": 0,
        #     "total_loss":0,
        #     "tp": np.zeros(self.num_classes, dtype=np.int64),
        #     "fp": np.zeros(self.num_classes, dtype=np.int64),
        #     "fn": np.zeros(self.num_classes, dtype=np.int64),
        #     "pred_flatten": [],
        #     "label_flatten": []
        # }
        # total_metrics中每个元素为
        # {
        #             "avg_loss": avg_loss,
        #             "oa": oa,
        #             "mpa": mpa,
        #             "miou": miou,
        #             "precision": precision,
        #             "recall": recall,
        #             "f1": f1,
        #             "kappa": kappa
        #         }

        # 训练集
        # todo 这个只能所有用户都参与, 如果是挑选部分, 这个就不行了
        ids, num_samples, total_metrics, total_stats = self.train_metrics()
        print("\n")
        total_samples = sum(num_samples)
        w = np.array(num_samples) / total_samples  # 每个客户端的权重
        # 遍历提取值
        metrics_keys = total_metrics[0].keys()
        metrics_dict = {key: [] for key in metrics_keys}
        for metric in total_metrics:
            for key in metrics_keys:
                metrics_dict[key].append(metric.get(key, None))

        for key in metrics_keys:
            avg, avg_std = (np.array(metrics_dict[key]) * w).sum(), (np.array(metrics_dict[key]) * w).std()
            print(f"{'Train avg_'+key:<28}: {avg:.9f}")
            print(f"{'Train avg_'+key+'_std':<28}: {avg_std:.9f}")
            if f"train_avg_{key}" not in self.per_round_metrics_results.keys():
                self.per_round_metrics_results[f"train_avg_{key}"] = []
            self.per_round_metrics_results[f"train_avg_{key}"].append(avg)

            if f"train_avg_{key}_std" not in self.per_round_metrics_results.keys():
                self.per_round_metrics_results[f"train_avg_{key}_std"] = []
            self.per_round_metrics_results[f"train_avg_{key}_std"].append(avg_std)
        print("\n")
        # 测试集
        ids, num_samples, total_metrics, total_stats = self.test_metrics()
        print("\n")
        total_samples = sum(num_samples)
        w = np.array(num_samples) / total_samples  # 每个客户端的权重
        # 遍历提取值
        metrics_keys = total_metrics[0].keys()
        metrics_dict = {key: [] for key in metrics_keys}
        for metric in total_metrics:
            for key in metrics_keys:
                metrics_dict[key].append(metric.get(key, None))
        for key in metrics_keys:
            avg, avg_std = (np.array(metrics_dict[key]) * w).sum(), (np.array(metrics_dict[key]) * w).std()
            print(f"{'Test avg_'+key:<28}: {avg:.9f}")
            print(f"{'Test avg_'+key+'_std':<28}: {avg_std:.9f}")
            if f"test_avg_{key}" not in self.per_round_metrics_results.keys():
                self.per_round_metrics_results[f"test_avg_{key}"] = []
            self.per_round_metrics_results[f"test_avg_{key}"].append(avg)

            if f"test_avg_{key}_std" not in self.per_round_metrics_results.keys():
                self.per_round_metrics_results[f"test_avg_{key}_std"] = []
            self.per_round_metrics_results[f"test_avg_{key}_std"].append(avg_std)

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        total_metrics = []
        total_stats = []
        for c in self.clients:
            total_samples, metrics, base_stats = c.test_metrics()
            num_samples.append(total_samples)
            total_metrics.append(metrics)
            total_stats.append(base_stats)

        ids = [c.id for c in self.clients]

        return ids, num_samples, total_metrics, total_stats

    def train_metrics(self):
        # if self.eval_new_clients and self.num_new_clients > 0:
        #     return [0], [1], [0]
        num_samples = []
        total_metrics = []
        total_stats = []
        for c in self.clients:
            total_samples, metrics, base_stats = c.train_metrics()
            num_samples.append(total_samples)
            total_metrics.append(metrics)
            total_stats.append(base_stats)

        ids = [c.id for c in self.clients]

        return ids, num_samples, total_metrics, total_stats
