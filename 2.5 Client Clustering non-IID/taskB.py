from collections import OrderedDict
from logging import INFO
import time  # Added time to measure training times
import os
import hashlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

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

def load_data():
    """Load CIFAR-10 dataset and partition it non-IID among clients using Dirichlet distribution."""
    # Get the client ID
    CLIENT_ID = os.getenv("HOSTNAME")
    if CLIENT_ID is None:
        CLIENT_ID = "clientb-1"  # Default client ID

    # Map CLIENT_ID to client_index
    # Assuming CLIENT_ID is in the format "clientb-1", "clientb-2", etc.
    client_num_str = ''.join(filter(str.isdigit, CLIENT_ID))
    if client_num_str:
        client_index = int(client_num_str) - 1  # clientb-1 => 0
    else:
        client_index = 0

    num_clients = 2  # Adjust this if you have more clients

    # Set the random seed for consistent partitioning across clients
    np.random.seed(42)

    # Load the full CIFAR-10 dataset
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    full_trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    # Perform Dirichlet partitioning
    alpha = 0.5  # Adjust alpha as needed; smaller alpha => more heterogeneity
    client_indices = non_iid_partition(full_trainset, num_clients, alpha)

    # Get the indices for this client
    indices = client_indices[client_index]

    # Create subset and DataLoader
    train_subset = Subset(full_trainset, indices)
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)

    # Use the full test set
    testloader = DataLoader(testset)

    return trainloader, testloader

def non_iid_partition(dataset, num_clients, alpha):
    """Partition dataset among clients using Dirichlet distribution."""
    num_classes = 10  # CIFAR-10 has 10 classes
    targets = np.array(dataset.targets)
    idxs = np.arange(len(targets))

    # For each class, get the indices
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

    # Sample Dirichlet distribution for each class
    proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients), size=num_classes)

    # For each client, collect indices
    client_indices = [[] for _ in range(num_clients)]
    for cls_indices, cls_proportions in zip(class_indices, proportions):
        # Shuffle class indices
        np.random.shuffle(cls_indices)
        # Split indices according to sampled proportions
        splits = np.array_split(cls_indices, (np.cumsum(cls_proportions)[:-1] * len(cls_indices)).astype(int))
        for idx, split in enumerate(splits):
            client_indices[idx].extend(split)

    return client_indices

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

    train_loss, train_acc, train_f1 = test(net, trainloader)
    val_loss, val_acc, val_f1 = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
    }

    return results, training_time

def test(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0.0
    all_preds = []
    all_labels = []

    net.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    accuracy = correct / len(testloader.dataset)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate F1 score
    f1 = f1_score_torch(all_labels, all_preds, num_classes=10, average='macro')

    return loss, accuracy, f1

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate precision and recall for each class
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_per_class = torch.zeros(num_classes)
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP

        precision[i] = TP / (TP + FP + 1e-8)
        recall[i] = TP / (TP + FN + 1e-8)
        f1_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)

    if average == 'macro':
        f1 = f1_per_class.mean().item()
    elif average == 'micro':
        TP = torch.diag(confusion_matrix).sum()
        FP = confusion_matrix.sum() - torch.diag(confusion_matrix).sum()
        FN = FP
        precision_micro = TP / (TP + FP + 1e-8)
        recall_micro = TP / (TP + FN + 1e-8)
        f1 = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)).item()
    else:
        raise ValueError("Average must be 'macro' or 'micro'")

    return f1

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)