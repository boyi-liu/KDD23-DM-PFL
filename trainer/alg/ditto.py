import copy

import torch

from models.config import load_model
from trainer.base import BaseServer, BaseClient

def add_args(parser):
    parser.add_argument('--p_epoch', type=int, default=5, help="Epoch for personalized part")
    parser.add_argument('--lam', type=float, default=0.1, help="Lambda")
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.lam = args.lam
        self.p_epoch = args.p_epoch
        self.p_model = load_model(args).to(args.device)
        self.p_optim = copy.deepcopy(self.optim)


    def run(self):
        self.train()
        self.p_train()


    def p_train(self):
        self.p_model.train()
        gm = self.model2tensor()
        # === train ===
        batch_loss = []
        for epoch in range(self.p_epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.p_optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.p_model(image)

                pm = torch.cat([p.view(-1) for p in self.p_model.parameters()], dim=0)
                loss = self.loss_func(predict_label, label) + self.lam/2 * torch.norm(gm - pm, p=2)
                loss.backward()
                self.p_optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        # self.metric['p_loss'].append(sum(batch_loss) / len(batch_loss))

    def local_test(self, g_test=False):
        if not g_test:
            self.model.load_state_dict(self.p_model.state_dict())
        return super().local_test()

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        self.optim = torch.optim.SGD(params=self.model.parameters(),
                                     lr=(self.lr * (self.args.gamma ** self.server.round)))
        self.p_optim = torch.optim.SGD(params=self.p_model.parameters(),
                                       lr=(self.lr * (self.args.gamma ** self.server.round)))


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()