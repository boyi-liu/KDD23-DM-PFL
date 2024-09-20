import copy
import numpy as np
import torch

from trainer.base import BaseServer, BaseClient
from utils.prune_util import param_prune, param_regrow, param_prune_to_sparsity


def add_args(parser):
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity')
    parser.add_argument('--readjust_ratio', type=float, default=0.01, help='Proportion of readjust')
    parser.add_argument('--readjust_epoch', type=int, default=3, help='Which epoch to readjust')
    parser.add_argument('--readjust_round_gap', type=float, default=10, help='How many rounds between readjustment')
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.mask = {name: torch.ones_like(param, dtype=torch.int) for name, param in self.model.named_parameters()}

        self.readjust_epoch = args.readjust_epoch  # the given epoch for readjust
        self.alpha = args.readjust_ratio  # the basic readjust ratio
        self.sparsity = args.sparsity  # the given sparsity
        self.readjust_round_gap = args.readjust_round_gap

    def run(self):
        self.train()

    def cosine_annealing(self):
        r = self.server.round
        R_end = self.args.rnd
        return self.alpha / 2 * (1 + np.cos(r * np.pi / R_end))

    def apply_mask(self):
        print(self.mask.keys())
        for name, param in self.model.named_parameters():
            param.data *= self.mask[name]

    def train(self):
        batch_loss = []
        readjust = ((self.server.round - 1) % self.readjust_round_gap) == 0
        self.apply_mask()
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

                self.apply_mask()
            # NOTE: extra prune is needed
            if epoch == self.readjust_epoch and readjust:
                temp_mask, n_prune_dict = param_prune(model=self.model,
                                                      mask=self.mask,
                                                      prune_proportion=self.cosine_annealing())
                self.mask = param_regrow(model=self.model,
                                         mask=temp_mask,
                                         n_prune_dict=n_prune_dict)

        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.mask = {name: torch.ones_like(param, dtype=torch.int) for name, param in self.model.named_parameters()}
        self.sparsity = args.sparsity  # the given sparsity
        self.min_votes = 0.2

        self.received_masks = []

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)
            client.mask = copy.deepcopy(self.mask)

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        super().uplink()
        self.received_masks = [self.masks_to_tensor() * client.weight
                               for client in self.sampled_clients]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)

        # === aggregate ===
        avg_params = sum(self.received_params)
        avg_masks = sum(self.received_masks)

        # === fine-tune masks ===
        avg_masks = torch.where(avg_masks > self.min_votes, 1, 0)
        avg_params /= avg_masks
        avg_params = torch.nan_to_num(avg_params, nan=0.0, posinf=0.0, neginf=0.0)

        # === load into model & mask ===
        self.tensor2model(avg_params)
        self.tensor_to_masks(avg_masks)

        # === prune ===
        self.mask = param_prune_to_sparsity(model=self.model, prune_sparsity=self.sparsity)

    def masks_to_tensor(self):
        return torch.cat([m.view(-1) for m in self.mask.values()], dim=0)

    def tensor_to_masks(self, tensor):
        m_index = 0
        for mname, m in self.mask.items():
            m_size = m.numel()
            self.mask[mname] = tensor[m_index: m_index + m_size].view(m.shape).detach().clone()
            m_index += m_size
