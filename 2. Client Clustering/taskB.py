from collections import OrderedDict
from logging import INFO
import time  
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

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

def count_classes_subset(dataset, subset_indices):
    counts = {i: 0 for i in range(10)}
    for idx in subset_indices:
        _, label = dataset[idx]
        counts[label] += 1
    return counts

def load_data(client_id, alpha=0.5):
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    # Costruisce un dizionario di indici per ogni classe
    class_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        class_to_indices[label].append(idx)

    # Distribuzione Dirichlet per i campioni per classe
    num_classes = 10
    total_samples = 50000
    proportions = np.random.dirichlet([alpha] * num_classes)  # Distribuzione Dirichlet
    class_counts = [int(p * total_samples) for p in proportions]  # Campioni per classe
    
    selected_indices = []
    for cls, count in enumerate(class_counts):
        available_indices = class_to_indices[cls]
        selected = random.sample(available_indices, min(count, len(available_indices)))
        selected_indices.extend(selected)

    # Creazione del subset del training set e DataLoader
    subset_train = Subset(trainset, selected_indices)
    trainloader = DataLoader(subset_train, batch_size=32, shuffle=True)
    
    # Creazione del DataLoader per il test set
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    actual_class_counts = count_classes_subset(trainset, selected_indices)
    print(f"Task B - Client {client_id} - Distribuzione classi nel training set:")
    for cls_idx, count in actual_class_counts.items():
        print(f"  {CLASS_NAMES[cls_idx]}: {count} campioni")
    
    log(INFO, f"Client {client_id} - Distribuzione classi: {class_counts}")

    return trainloader, testloader

def train(net, trainloader, valloader, epochs, device):

    log(INFO, "Starting training...")

    # Start measuring training time
    start_time = time.time()

    net.to(device)  # sposta il modello sulla GPU se disponibile
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

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_preds.append(predicted)
            all_labels.append(labels)

    accuracy = correct / len(testloader.dataset)

    # Concatenare tutte le predizioni e le etichette
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calcolo dell'F1 score
    f1 = f1_score_torch(all_labels, all_preds, num_classes=10, average='macro')

    return loss, accuracy, f1


def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    # Creazione della matrice di confusione
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1

    # Calcolo di precision e recall per ogni classe
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