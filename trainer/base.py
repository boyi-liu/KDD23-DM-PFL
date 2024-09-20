import importlib

import torch
import torch.nn as nn
import time
import random

from utils.data_utils import read_client_data
from models.config import load_model
from utils.dataprocess import DataProcessor
from torch.utils.data import DataLoader


class BaseClient:
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset_train = read_client_data(args.dataset, self.id, is_train=True)
        self.dataset_test = read_client_data(args.dataset, self.id, is_train=False)
        self.device = args.device
        self.server = None

        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        self.model = load_model(args).to(args.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)

        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
        }

        if self.dataset_train is not None:
            self.loader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None
            )
        if self.dataset_test is not None:
            self.loader_test = DataLoader(
                dataset=self.dataset_test,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None
            )

        self.p_params = [False for _ in self.model.parameters()]  # default: all global, no personalized

        self.training_time = None
        self.lag_level = args.lag_level
        self.weight = 1

    def run(self):
        raise NotImplementedError()

    def train(self):
        # === train ===
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

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def clone_model(self, target):
        p_tensor = target.model2tensor()
        self.tensor2model(p_tensor)

    def local_test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.loader_test:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.00 * correct / total

        self.metric['acc'].append(acc)

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        self.optim = torch.optim.SGD(params=self.model.parameters(),
                                     lr=(self.lr * (self.args.gamma ** self.server.round)))

    def model2tensor(self):
        return torch.cat([p.data.view(-1) for idx, p in enumerate(self.model.parameters())
                          if self.p_params[idx] is False], dim=0)

    def tensor2model(self, tensor):
        param_index = 0
        for is_p, param in zip(self.p_params, self.model.parameters()):
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



class BaseServer(BaseClient):
    def __init__(self, id, args, clients):
        super().__init__(id, args)
        self.client_num = args.total_num
        self.sample_rate = args.sr
        self.clients = clients
        self.sampled_clients = []
        self.total_round = args.rnd

        self.round = 0
        self.wall_clock_time = 0

        self.received_params = []

        for client in self.clients:
            client.server = self

        self.TO_LOG = True

    def run(self):
        raise NotImplementedError()

    def sample(self):
        sample_num = int(self.sample_rate * self.client_num)
        self.sampled_clients = random.sample(self.clients, sample_num)

        total_samples = sum(len(client.dataset_train) for client in self.sampled_clients)
        for client in self.sampled_clients:
            client.weight = len(client.dataset_train) / total_samples

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)

    def client_update(self):
        for client in self.sampled_clients:
            client.model.train()
            client.reset_optimizer()
            start_time = time.time()

            client.run()

            end_time = time.time()
            client.training_time = (end_time - start_time) * client.lag_level
        self.wall_clock_time += max([client.training_time for client in self.sampled_clients])

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = [client.model2tensor() * client.weight
                                for client in self.sampled_clients]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        avg_tensor = sum(self.received_params)
        self.tensor2model(avg_tensor)

    def test_all(self):
        for client in self.clients:
            c_metric = client.metric
            if client in self.sampled_clients:
                self.metric['loss'].append(c_metric['loss'].last())

            client.clone_model(self)
            client.local_test()

            self.metric['acc'].append(c_metric['acc'].last())
        return self.analyse_metric()

    def analyse_metric(self):
        acc = self.metric['acc'].avg()
        loss = self.metric['loss'].avg()
        std = self.metric['acc'].std()

        self.metric['acc'].clear()
        self.metric['loss'].clear()

        return {'loss': loss,
                'acc': acc,
                'std': std}