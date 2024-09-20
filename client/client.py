import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import flwr as fl
from flwr.client import NumPyClient
from collections import OrderedDict
from typing import List, Dict
import os

# Imposta il dispositivo
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_datasets(partition_id, num_partitions: int):
    # Definisci le trasformazioni
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Carica il dataset CIFAR-10
    cifar10_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
    cifar10_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Suddividi il dataset tra i client
    total_size = len(cifar10_train)
    partition_size = total_size // num_partitions
    indices = list(range(total_size))
    partition_indices = indices[partition_id * partition_size:(partition_id + 1) * partition_size]

    train_subset = torch.utils.data.Subset(cifar10_train, partition_indices)
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)

    valloader = DataLoader(cifar10_test, batch_size=32)

    return trainloader, valloader

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Input channels = 3 for RGB images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted for CIFAR-10 image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10 has 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def main():
    # Il partition_id e num_partitions dovrebbero essere passati come variabili d'ambiente
    partition_id = int(os.environ.get("PARTITION_ID", "0"))
    num_partitions = int(os.environ.get("NUM_CLIENTS", "2"))

    net = Net().to(DEVICE)
    trainloader, valloader = load_datasets(partition_id, num_partitions)
    client = FlowerClient(partition_id, net, trainloader, valloader)
    fl.client.start_numpy_client(server_address="server:8080", client=client)

if __name__ == "__main__":
    main()
