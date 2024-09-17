import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
import flwr as fl
from flwr.client import NumPyClient
from torchvision import transforms
from flwr_datasets import FederatedDataset
from typing import List

# Definizione del modello
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Caricamento del dataset
def load_datasets(partition_id: int, num_partitions: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, valloader

# Definizione del client
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(1)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def train(self, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        for epoch in range(epochs):
            for batch in self.trainloader:
                images, labels = batch["img"], batch["label"]
                images, labels = images.to("cpu"), labels.to("cpu")
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def test(self):
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.net.eval()
        with torch.no_grad():
            for batch in self.valloader:
                images, labels = batch["img"], batch["label"]
                images, labels = images.to("cpu"), labels.to("cpu")
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return loss / total, correct / total

# Funzione per la creazione dei client
def client_fn(context):
    partition_id = context.node_config["partition-id"]
    net = Net().to("cpu")
    trainloader, valloader = load_datasets(partition_id, NUM_PARTITIONS)
    return FlowerClient(partition_id, net, trainloader, valloader)
