import os
import time

import h5py
import numpy as np
from flcore.myclients.myclientavg import LoveDA2021RuralClient
from flcore.servers.serverbase import Server


class LoveDA2021RuralFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(LoveDA2021RuralClient)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.rs_train_avg_loss = []
        self.rs_train_std_loss = []
        self.rs_train_avg_IoU = []
        self.rs_train_std_IoU = []

        self.rs_test_avg_loss = []
        self.rs_test_std_loss = []
        self.rs_test_avg_IoU = []
        self.rs_test_std_IoU = []

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            client = clientObj(self.args, id=i, )
            self.clients.append(client)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            client = clientObj(self.args, id=i)
            self.clients.append(client)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # 服务端本地评估效果
            if i % self.eval_gap == 0:
                print(f"\nRound number: {i}", "-" * 20)
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('time cost:', self.Budget[-1], '-' * 25)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

        # self.rs_train_avg_loss = []
        # self.rs_train_std_loss = []
        # self.rs_train_avg_IoU = []
        # self.rs_train_std_IoU = []
        #
        # self.rs_test_avg_loss = []
        # self.rs_test_std_loss = []
        # self.rs_test_avg_IoU = []
        # self.rs_test_std_IoU = []
        print("\nBest Test IoU:", max(self.rs_test_avg_IoU))
        print("\nAverage time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(LoveDA2021RuralClient)
        #     print(f"\nFine tuning round", "-" * 20)
        #     print("\nEvaluate new clients")
        #     self.evaluate()

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)
            # todo 不好用, 改为json吧
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset("rs_train_avg_loss", data=self.rs_train_avg_loss)
                hf.create_dataset("rs_train_std_loss", data=self.rs_train_std_loss)
                hf.create_dataset("rs_train_avg_IoU", data=self.rs_train_avg_IoU)
                hf.create_dataset("rs_train_std_IoU", data=self.rs_train_std_IoU)
                hf.create_dataset("rs_test_avg_loss", data=self.rs_test_avg_loss)
                hf.create_dataset("rs_test_std_loss", data=self.rs_test_std_loss)
                hf.create_dataset("rs_test_avg_IoU", data=self.rs_test_avg_IoU)
                hf.create_dataset("rs_test_std_IoU", data=self.rs_test_std_IoU)

    def evaluate(self, acc=None, loss=None):
        # 对训练集进行评估
        ids, num_samples, total_losses, total_IoUs = self.train_metrics()
        train_avg_loss, train_avg_IoU = sum(total_losses) / sum(num_samples), sum(total_IoUs) / sum(num_samples)
        print("Averaged Train Loss: {:.6f}".format(train_avg_loss))
        print("Averaged Train IoU: {:.6f}".format(train_avg_IoU))
        print("Std Train Loss: {:.6f}".format(np.std(np.array(total_losses) / np.array(num_samples))))
        print("Std Train IoU: {:.6f}".format(np.std(np.array(total_IoUs) / np.array(num_samples))))

        self.rs_train_avg_loss.append(train_avg_loss)
        self.rs_train_std_loss.append(train_avg_IoU)
        self.rs_train_avg_IoU.append(np.std(np.array(total_losses) / np.array(num_samples)))
        self.rs_train_std_IoU.append(np.std(np.array(total_IoUs) / np.array(num_samples)))

        # 对测试集进行评估
        ids, num_samples, total_losses, total_IoUs = self.test_metrics()
        test_avg_loss, test_avg_IoU = sum(total_losses) / sum(num_samples), sum(total_IoUs) / sum(num_samples)
        print("Averaged Train Loss: {:.6f}".format(test_avg_loss))
        print("Averaged Train IoU: {:.6f}".format(test_avg_IoU))
        print("Std Train Loss: {:.6f}".format(np.std(np.array(total_losses) / np.array(num_samples))))
        print("Std Train IoU: {:.6f}".format(np.std(np.array(total_IoUs) / np.array(num_samples))))

        self.rs_test_avg_loss.append(test_avg_loss)
        self.rs_test_std_loss.append(test_avg_IoU)
        self.rs_test_avg_IoU.append(np.std(np.array(total_losses) / np.array(num_samples)))
        self.rs_test_std_IoU.append(test_avg_IoU)

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        total_losses = []
        total_IoUs = []
        for c in self.clients:
            test_num, total_loss, total_IoU = c.test_metrics()
            total_losses.append(total_loss)
            total_IoUs.append(total_IoU)
            num_samples.append(test_num)

        ids = [c.id for c in self.clients]

        return ids, num_samples, total_losses, total_IoUs

    def train_metrics(self):
        # if self.eval_new_clients and self.num_new_clients > 0:
        #     return [0], [1], [0]
        num_samples = []
        total_losses = []
        total_IoUs = []
        for c in self.clients:
            test_num, total_loss, total_IoU = c.train_metrics()
            total_losses.append(total_loss)
            total_IoUs.append(total_IoU)
            num_samples.append(test_num)

        ids = [c.id for c in self.clients]

        return ids, num_samples, total_losses, total_IoUs
