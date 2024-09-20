from trainer.base import BaseServer, BaseClient

def add_args(parser):
    parser.add_argument('--ft_epoch', type=int, default=1, help='How many local epochs for finetune')
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.ft_epoch = args.ft_epoch

    def run(self):
        self.finetune()
        self.local_test()
        self.train()

    def finetune(self):
        for epoch in range(self.ft_epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()

    def train(self):
        old_epoch = self.epoch
        self.epoch -= self.ft_epoch
        super().train()
        self.epoch = old_epoch


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def test_all(self):
        for client in self.clients:
            c_metric = client.metric
            if client in self.sampled_clients:
                self.metric['loss'].append(client.metric['loss'].last())
            # NOTE: acc = 0 means that the client is not sampled yet
            if not c_metric['acc'].is_empty():
                self.metric['acc'].append(c_metric['acc'].last())
        return self.analyse_metric()