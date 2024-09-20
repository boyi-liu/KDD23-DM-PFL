from trainer.base import BaseServer, BaseClient
from enum import Enum
from utils.prune_util import param_prune, param_regrow, param_prune_to_sparsity

import copy
import numpy as np
import torch

def add_args(parser):
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity')
    parser.add_argument('--readjust_ratio', type=float, default=0.7, help='Proportion of readjust')
    parser.add_argument('--readjust_epoch', type=int, default=3, help='Which epoch to readjust')
    parser.add_argument('--readjust_round_gap', type=float, default=3, help='How many rounds between readjustment')
    parser.add_argument('--iter', type=int, default=2, help='How many iterations')
    return parser.parse_args()

class Phase(Enum):
    Default = 0
    Dual = 1
    Generic = 2
    Personalized = 3

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)

        self.global_mask = {name: torch.ones_like(param, dtype=torch.int) for name, param in self.model.named_parameters()}
        self.local_mask = {name: torch.ones_like(param, dtype=torch.int) for name, param in self.model.named_parameters()}

        self.global_state = copy.deepcopy(self.model.state_dict())
        self.local_state = copy.deepcopy(self.model.state_dict())

        self.readjust_epoch = args.readjust_epoch  # the given epoch for readjust
        self.readjust_round_gap = args.readjust_round_gap
        self.sparsity = args.sparsity  # the given sparsity
        self.alpha = args.readjust_ratio

        self.sampled = False

        self.diff_rate = 0

    def run(self):
        phase = self.server.phase
        if phase == Phase.Dual:
            self.set_local_overlap()
            self.train_dual_mask()
            self.local_state = copy.deepcopy(self.model.state_dict())
        elif phase == Phase.Generic:
            self.model.load_state_dict(self.global_state)
            self.train_generic() # NOTE: no need to update local state
        else:
            if not self.sampled:
                self.set_local_overlap(zero=True)  # set un-overlapped part to zero
                self.sampled = True
            self.train_personalized()
            self.local_state = copy.deepcopy(self.model.state_dict())

    def cosine_annealing(self):
        r = self.server.round
        R_end = self.args.rnd
        return self.alpha / 2 * (1 + np.cos(r * np.pi / R_end))

    def train_dual_mask(self):
        readjust = ((self.server.round - 1) % self.readjust_round_gap) == 0


        self.apply_mask_weight(mode='c')
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()

                self.apply_mask_weight(mode='c')

                batch_loss.append(loss.item())

            # === Refine local mask ===
            if epoch == self.readjust_epoch and readjust:
                temp_mask, n_prune_dict = param_prune(model=self.model, mask=self.local_mask, prune_proportion=self.cosine_annealing())
                self.local_mask = param_regrow(model=self.model, mask=temp_mask, n_prune_dict=n_prune_dict)
                self.apply_mask_weight(mode='c')  # NOTE: not sure whether we need it?

        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def train_generic(self):
        self.apply_mask_weight(mode='g')

        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())
                self.apply_mask_weight(mode='g')

        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def train_personalized(self):
        self.apply_mask_weight(mode='c')
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())
                self.apply_mask_weight(mode='d')

        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


    def set_local_overlap(self, zero=False):
        """
        Change {m_g \cap m_c} part of local model
        :param zero: if we set the un-overlapped part to zero
        """

        for name, param in self.model.named_parameters():
            m_g = self.global_mask[name].bool()
            m_c = self.local_mask[name].bool()

            overlap = m_g & m_c
            self.local_state[name][overlap] = self.global_state[name][overlap]
            if zero:
                self.local_state[name][m_c & (~overlap)] = 0

        self.model.load_state_dict(self.local_state)

    def apply_mask_weight(self, mode='c'):
        """
               |-- 'c' -> apply local mask
        Mode --|-- 'g' -> apply global mask
               |-- 'd' -> referred to different/dual, apply different part of local and global mask
        """
        for name, param in self.model.named_parameters():
            if mode == 'c':
                param.data *= self.local_mask[name]
            elif mode == 'g':
                param.data *= self.global_mask[name]
            elif mode == 'd':
                m_g, m_c = self.global_mask[name].bool(), self.local_mask[name].bool()
                overlap = (m_g & m_c).bool()
                param.data *= m_c
                param.data[overlap] = self.global_state[name][overlap]

    def theta_c(self):
        theta_c = copy.deepcopy(self.local_state)
        for name, param in self.model.named_parameters():
            m_g = self.global_mask[name]
            m_c = self.local_mask[name]
            overlap = (m_g & m_c).bool()
            local_all = m_c.bool()

            theta_c[name][overlap] = self.global_state[name][overlap]
            theta_c[name][~local_all] = 0
        return theta_c

    def theta_g(self):
        theta_g = copy.deepcopy(self.global_state)
        for name, param in self.model.named_parameters():
            m_g = self.global_mask[name]
            global_all = m_g.bool()
            theta_g[name][~global_all] = 0
        return theta_g

    def local_test(self, g_test=False):
        if g_test:
            self.model.load_state_dict(self.theta_g())
        else:
            self.model.load_state_dict(self.theta_c())

        super().local_test()


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.phase = Phase.Default
        self.global_mask = {name: torch.ones_like(param, dtype=torch.int) for name, param in self.model.named_parameters()}

        self.iteration_num = args.iter  # how many total iterations
        self.sparsity = args.sparsity  # the given sparsity
        self.min_votes = 0.5  # the bound to keep a mask during mask-aggregation

        self.init_dual_mask()
        self.received_masks = []

    def init_dual_mask(self):
        self.global_mask = param_prune_to_sparsity(model=self.model, prune_sparsity=self.sparsity)
        for name, param in self.model.named_parameters():
            param.data *= self.global_mask[name]

        for client in self.clients:
            client.local_mask = copy.deepcopy(self.global_mask)
            client.apply_mask_weight('c')


    def set_phase(self):
        """
        Set current phase firstly, based on current global round
        :return:
        """
        rnd = self.round
        itn = self.iteration_num  # how many iterations
        its = self.total_round / itn  # how many rounds in an iteration, notes as 'iteration size'

        dual_trigger_rnd = [int(its * _) for _ in range(itn)]
        generic_trigger_rnd = [int(_ + its / 2) for _ in dual_trigger_rnd]
        personalized_trigger_rnd = [int(_ + its / 4 * 3) for _ in dual_trigger_rnd]

        if self.round == 0:
            print('dual', dual_trigger_rnd)
            print('generic', generic_trigger_rnd)
            print('personalized', personalized_trigger_rnd)

        if rnd in dual_trigger_rnd:
            self.phase = Phase.Dual
        if rnd in generic_trigger_rnd:
            self.phase = Phase.Generic
        if rnd in personalized_trigger_rnd:
            self.phase = Phase.Personalized
            for c in self.clients:
                c.sampled = False

    def run(self):
        self.set_phase()
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def downlink(self):
        """
        We extra need to downlink the global mask here
        There is no need to copy.deepcopy the global mask here
        """
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.global_state = self.model.state_dict()
            client.global_mask = self.global_mask

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        super().uplink()
        if self.phase == Phase.Dual:
            self.received_masks = [masks_to_tensor(mask=client.local_mask) * client.weight
                                   for client in self.sampled_clients]

    def aggregate(self):
        # NOTE: only 1) Dual- and 2) Generic- phase will aggregate weight
        # NOTE: only Dual-phase will aggregate mask
        assert (len(self.sampled_clients) > 0)

        if self.phase == Phase.Personalized:
            return
        avg_params = sum(self.received_params)
        if self.phase == Phase.Dual:
            avg_masks = sum(self.received_masks)
            avg_masks = torch.where(avg_masks > self.min_votes, 1, 0)
            avg_params /= avg_masks
            avg_params = torch.nan_to_num(avg_params, nan=0.0, posinf=0.0, neginf=0.0)
            self.global_mask = tensor_to_masks(tensor=avg_masks, mask=self.global_mask)
        self.tensor2model(avg_params)
        self.global_mask = param_prune_to_sparsity(model=self.model, prune_sparsity=self.sparsity)


def masks_to_tensor(mask):
    return torch.cat([m.view(-1) for m in mask.values()], dim=0)

def tensor_to_masks(tensor, mask):
    m_index = 0
    new_mask = copy.deepcopy(mask)
    for mname, m in mask.items():
        m_size = m.numel()
        new_mask[mname] = tensor[m_index: m_index+m_size].view(m.shape).detach().clone()
        m_index += m_size
    return new_mask