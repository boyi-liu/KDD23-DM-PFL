import torch

from trainer.base import BaseServer, BaseClient


def add_args(parser):
    parser.add_argument('--mu', type=float, default=0.05, help="Mu")
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.mu = args.mu

    def train(self):
        gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)

        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)

                pm = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0)
                loss = self.loss_func(predict_label, label) + self.mu * torch.norm(gm - pm, p=2)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def run(self):
        self.train()


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
