import flwr as fl
import torch
import time
import numpy as np

class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def main():
    model = DummyModel()

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self):
            start_time = time.time()
            params = [val.cpu().numpy() for val in model.parameters()]
            communication_time = time.time() - start_time
            return params, communication_time

        def set_parameters(self, parameters):
            for p, new_p in zip(model.parameters(), parameters):
                p.data = torch.from_numpy(new_p)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            start_time = time.time()
            # Simula l'addestramento
            for _ in range(10):
                for param in model.parameters():
                    param.data += torch.randn_like(param)
            training_time = time.time() - start_time
            params, communication_time = self.get_parameters()
            
            # Invia al server i tempi di training e comunicazione
            return params, len(model.parameters()), {"training_time": training_time, "communication_time": communication_time}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = np.random.random()
            return loss, len(model.parameters()), {}

    fl.client.start_numpy_client(server_address="server:8080", client=FlowerClient())

if __name__ == "__main__":
    main()
