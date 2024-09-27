# server.py

from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imposta il backend non interattivo di matplotlib
matplotlib.use('Agg')

# Configurazione del dispositivo
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

# Funzione per ottenere i pesi del modello
def get_weights(net):
    # Assicura l'ordinamento consistente delle chiavi
    state_dict = net.state_dict()
    ordered_state_dict = OrderedDict(sorted(state_dict.items()))
    return [val.cpu().numpy() for key, val in ordered_state_dict.items()]

# Variabile globale per tenere traccia del round corrente
currentRnd = 0
num_rounds = 3  # Numero totale di round

# Ottieni il percorso assoluto della directory corrente
current_dir = os.path.abspath(os.path.dirname(__file__))

# Crea la directory per i log delle performance
performance_dir = os.path.join(current_dir, 'performance')
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Definisci il percorso del file CSV
csv_file = os.path.join(performance_dir, 'performance.csv')

# Inizializza il file CSV, sovrascrivendolo se esiste
if os.path.exists(csv_file):
    try:
        os.remove(csv_file)
        print(f"File '{csv_file}' rimosso con successo.")
    except OSError as e:
        print(f"Errore nella rimozione del file: {e}")

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time'])

# Funzione per generare i grafici delle performance
def generate_performance_graphs():
    # (Il codice per generare i grafici rimane invariato)
    pass

# Definisci la funzione di aggregazione delle metriche
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global currentRnd  # Dichiarazione esplicita della variabile globale

    examples = [num_examples for num_examples, _ in metrics]

    # Moltiplica la precisione di ogni client per il numero di esempi usati
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    currentRnd += 1

    # Se siamo nell'ultimo round, genera i grafici
    if currentRnd == num_rounds:
        generate_performance_graphs()

    # Aggrega e ritorna le metriche personalizzate (media ponderata)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }

# Inizializza i parametri del modello
net = Net()
ndarrays = get_weights(net)
parameters = ndarrays_to_parameters(ndarrays)

if __name__ == "__main__":
    from flwr.server import start_server

    # Definisci la strategia personalizzata
    strategy = FedAvg(
        fraction_fit=1.0,  # Seleziona tutti i client disponibili
        fraction_evaluate=0.0,  # Disabilita la valutazione
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Avvia il server con la strategia personalizzata
    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
