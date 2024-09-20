from trainer.base import BaseClient, BaseServer


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_keys = ['conv2', 'fc']
        self.p_params = [name.split('.')[0] in self.local_keys
                         for name, _ in self.model.named_parameters()]

    def run(self):
        self.train()


# extend Client to get the self.p_params
class Server(BaseServer, Client):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()