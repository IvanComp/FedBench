# client.py

# Importa le librerie necessarie
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

# Definizione delle variabili e costanti
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definizione del modello
class Net(nn.Module):
    """Modello (simple CNN adattato da 'PyTorch: A 60 Minute Blitz')"""

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

# Funzioni per il caricamento dei dati e per il training/testing
def load_data():
    """Carica CIFAR-10 (set di training e test)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def train(net, trainloader, valloader, epochs, device):
    """Allena il modello sul set di training, misurando il tempo."""
    log(INFO, "Inizio training...")

    # Inizio misurazione tempo di training
    start_time = time.time()

    net.to(device)  # sposta il modello su GPU se disponibile
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

    # Fine misurazione tempo di training
    training_time = time.time() - start_time
    log(INFO, f"Training completato in {training_time:.2f} secondi")

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }

    return results, training_time  # Restituisce anche il tempo di training

def test(net, testloader):
    """Valida il modello sul set di test."""
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
    # Assicura l'ordinamento consistente delle chiavi
    state_dict = net.state_dict()
    ordered_state_dict = OrderedDict(sorted(state_dict.items()))
    return [val.cpu().numpy() for key, val in ordered_state_dict.items()]

def set_weights(net, parameters):
    # Assicura l'ordinamento consistente delle chiavi
    param_keys = [k for k, _ in sorted(net.state_dict().items())]
    print(f"Chiavi attese nel modello: {param_keys}")
    print(f"Numero di parametri attesi: {len(param_keys)}")
    print(f"Numero di parametri ricevuti: {len(parameters)}")

    if len(param_keys) != len(parameters):
        print("Errore: il numero di parametri ricevuti non corrisponde al numero di parametri attesi.")
        return

    # Crea lo state_dict dai parametri ricevuti
    params_dict = zip(param_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # Carica lo state_dict nel modello
    net.load_state_dict(state_dict, strict=True)

# Inizializzazione del client Flower
class FlowerClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        # Inizializza il modello e i data loader
        self.net = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data()

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid}: Inizio training...", flush=True)
        # Misura il tempo di comunicazione iniziale (ricezione dei parametri dal server)
        comm_start_time = time.time()

        # Setta i pesi e misura il tempo di training
        set_weights(self.net, parameters)
        results, training_time = train(self.net, self.trainloader, self.testloader, epochs=1, device=DEVICE)

        # Misura il tempo di comunicazione finale (completamento del ciclo di allenamento)
        comm_end_time = time.time()

        # Calcola il tempo di comunicazione
        communication_time = comm_end_time - comm_start_time

        # Logging del tempo di training
        print(f"CLIENT {self.cid}: Training completato in {training_time:.2f} secondi", flush=True)

        # Logging del tempo di comunicazione
        print(f"CLIENT {self.cid}: Tempo di comunicazione: {communication_time:.2f} secondi", flush=True)

        total_time = training_time + communication_time

        # Aggiunge i dati di timing al CSV (assicurati che il percorso sia corretto)
        csv_file = './performance/performance.csv'
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.cid, 0, training_time, communication_time, total_time])

        # Restituisce i pesi aggiornati, la dimensione dei dati di training e i risultati
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid}: Inizio valutazione...", flush=True)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        print(f"CLIENT {self.cid}: Valutazione completata", flush=True)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    original_cid = context.node_id
    original_cid_str = str(original_cid)
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]
    return FlowerClient(cid=cid).to_client()

# Flower ClientApp usando client_fn
app = ClientApp(client_fn=client_fn)

# Modalità legacy
if __name__ == "__main__":
    from flwr.client import start_client

    # Esempio di ID originale (puoi sostituirlo con il tuo metodo di generazione)
    original_cid = "1234567890"
    original_cid_str = str(original_cid)
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]

    # Avvia il client Flower
    start_client(
        server_address="server:8080",
        client=FlowerClient(cid=cid).to_client(),
    )
