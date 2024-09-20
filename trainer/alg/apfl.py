import copy
import numpy as np
import torch

from models.config import load_model
from trainer.base import BaseServer, BaseClient


def add_args(parser):
    parser.add_argument('--alpha', type=float, default=0.5, help="Aggregation parameter")
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.alpha = args.alpha  # follow the setup in page 14 of APFL
        self.p_model = load_model(args).to(args.device)
        self.p_optim = copy.deepcopy(self.optim)

    def run(self):
        self.train()
        self.personalization()

    def personalization(self):
        p_tensor = self.pmodel2tensor()
        g_tensor = self.model2tensor()
        aggr_tensor = (1 - self.alpha) * p_tensor + self.alpha * g_tensor
        self.tensor2pmodel(aggr_tensor)

    def pmodel2tensor(self):
        return torch.cat([p.data.view(-1) for idx, p in enumerate(self.p_model.parameters())
                          if self.p_params[idx] is False], dim=0)

    def tensor2pmodel(self, tensor):
        param_index = 0
        for is_p, param in zip(self.p_params, self.p_model.parameters()):
            if not is_p:
                # === get shape & total size ===
                shape = param.shape
                param_size = 1
                for s in shape:
                    param_size *= s

                # === put value into param ===
                # .clone() is a deep copy here
                param.data = tensor[param_index: param_index + param_size].view(shape).detach().clone()
                param_index += param_size

    def train(self):
        self.p_model.train()
        # === train ===
        batch_loss = []
        p_batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                # === global model ===
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

                # === personalized model ===
                self.p_optim.zero_grad()
                predict_label = self.p_model(image)
                p_loss = self.loss_func(predict_label, label)
                p_loss.backward()
                self.p_optim.step()
                p_batch_loss.append(p_loss.item())

                # === alpha ===
                self.alpha_update()

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    # https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.model.parameters(), self.p_model.parameters()):
            dif = p_params.data - l_params.data
            grad = self.alpha * p_params.grad.data + (1 - self.alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * self.alpha  # normalization
        self.alpha = self.alpha - self.lr * grad_alpha
        self.alpha = np.clip(self.alpha.item(), 0.0, 1.0)

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        self.optim = torch.optim.SGD(params=self.model.parameters(),
                                     lr=(self.lr * (self.args.gamma ** self.server.round)))
        self.p_optim = torch.optim.SGD(params=self.p_model.parameters(),
                                       lr=(self.lr * (self.args.gamma ** self.server.round)))

    def local_test(self, g_test=False):
        if not g_test:
            self.model.load_state_dict(self.p_model.state_dict())
        return super().local_test()

class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
