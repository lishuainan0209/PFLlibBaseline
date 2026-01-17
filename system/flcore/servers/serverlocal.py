import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\ntotal clients:{self.num_clients}, Join ratio:{self.join_ratio} ")
        print("Finished creating server and clients.")
        
        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\nRound number: {i}","-"*20)
                print("\nEvaluate personalized models")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.Budget.append(time.time() - s_t)
            print( f'time cost:{self.Budget[-1]:.4f}')

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(f"\nAverage time cost per round: {sum(self.Budget[1:]) / len(self.Budget[1:]):.4f} s")

        self.save_results()
        self.save_server_model()
