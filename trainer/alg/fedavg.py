from trainer.base import BaseServer, BaseClient

class Client(BaseClient):
    def run(self):
        self.train()


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
