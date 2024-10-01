# client.py

# Import necessary libraries
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import time
import csv
import os
import hashlib
from collections import OrderedDict
from logging import INFO
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

# Definition of variables and constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model definition
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Functions for loading data and for training/testing
def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set, measuring time."""
    log(INFO, "Starting training...")

    # Start measuring training time
    start_time = time.time()

    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    # End measuring training time
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }

    return results, training_time  # Also return the training time

def test(net, testloader):
    """Validate the model on the test set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def get_weights(net):
    # Ensure consistent key ordering
    state_dict = net.state_dict()
    ordered_state_dict = OrderedDict(sorted(state_dict.items()))
    return [val.cpu().numpy() for key, val in ordered_state_dict.items()]

def set_weights(net, parameters):
    # Ensure consistent key ordering
    param_keys = [k for k, _ in sorted(net.state_dict().items())]
    print(f"Expected keys in the model: {param_keys}")
    print(f"Expected number of parameters: {len(param_keys)}")
    print(f"Received number of parameters: {len(parameters)}")

    if len(param_keys) != len(parameters):
        print("Error: the number of received parameters does not match the expected number of parameters.")
        return

    # Create state_dict from received parameters
    params_dict = zip(param_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # Load state_dict into the model
    net.load_state_dict(state_dict, strict=True)

# Flower client initialization
class FlowerClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        # Initialize the model and data loaders
        self.net = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data()

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting training.", flush=True)

        # Monitoraggio iniziale della CPU
        cpu_start = psutil.cpu_percent(interval=None)  # Ottieni l'utilizzo della CPU prima del training

        comm_start_time = time.time()
        set_weights(net, parameters)
        results, training_time = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        comm_end_time = time.time()

        # Monitoraggio finale della CPU
        cpu_end = psutil.cpu_percent(interval=None)  # Ottieni l'utilizzo della CPU dopo il training

        # Calcola la differenza media di utilizzo della CPU
        cpu_usage = (cpu_start + cpu_end) / 2

        # Calcola i tempi
        communication_time = comm_end_time - comm_start_time
        total_time = training_time + communication_time

        # Append dati CPU e tempi al CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.cid, 0, training_time, communication_time, total_time, cpu_usage])

        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting evaluation...", flush=True)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        print(f"CLIENT {self.cid}: Evaluation completed", flush=True)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    original_cid = context.node_id
    original_cid_str = str(original_cid)
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]
    return FlowerClient(cid=cid).to_client()

# Flower ClientApp using client_fn
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    # Example of original ID (you can replace this with your own generation method)
    original_cid = "1234567890"
    original_cid_str = str(original_cid)
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]

    # Start the Flower client
    start_client(
        server_address="server:8080",
        client=FlowerClient(cid=cid).to_client(),
    )
