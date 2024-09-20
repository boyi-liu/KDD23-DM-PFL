from trainer.base import BaseServer, BaseClient

def add_args(parser):
    parser.add_argument('--p_epoch', type=int, default=5, help="Epoch for personalized part")
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.p_epoch = args.p_epoch
        self.local_keys = ['fc']
        self.p_params = [name.split('.')[0] in self.local_keys
                         for name, _ in self.model.named_parameters()]

    def run(self):
        self.train()

    def train(self):
        # === train ===
        batch_loss = []
        p_batch_loss = []

        # NOTE: freeze base, update head
        for idx, param in enumerate(self.model.parameters()):
            param.requires_grad = self.p_params[idx]

        for epoch in range(self.p_epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # NOTE: freeze head, update base
        for idx, param in enumerate(self.model.parameters()):
            param.requires_grad = not self.p_params[idx]

        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                p_batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

class Server(BaseServer, Client):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()