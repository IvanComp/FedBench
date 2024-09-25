from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from task import Net, get_weights
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# Imposta il backend non interattivo di matplotlib
matplotlib.use('Agg')

# Variabile globale per tenere traccia del round corrente
currentRnd = 0
num_rounds = 2  # Numero totale di round

# Crea la directory per i log delle performance
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Inizializza il file CSV, sovrascrivendolo
csv_file = performance_dir + 'performance.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)  # Sovrascrivi il file precedente

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'Training Time', 'Communication Time', 'Total Time'])

# Funzione per misurare e loggare il tempo di comunicazione
def measure_communication_time(start_time, end_time):
    communication_time = end_time - start_time
    print(f"Communication time: {communication_time:.2f} seconds")
    return communication_time

# Funzione per loggare il tempo di ogni round
def log_round_time(client_id, training_time, communication_time):
    total_time = training_time + communication_time
    print(f"CLIENT {client_id}: Round completed with total time {total_time:.2f} seconds")

    # Salva i dati nel CSV
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([client_id, training_time, communication_time, total_time])

# Funzione per generare i grafici delle performance
def generate_performance_graphs():
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))

    # Crea gli istogrammi per Training Time, Communication Time e Total Time
    df_melted = df.melt(id_vars=["Client ID"],
                        value_vars=["Training Time", "Communication Time", "Total Time"],
                        var_name="Metric",
                        value_name="Time (seconds)")

    sns.barplot(x="Metric", y="Time (seconds)", hue="Client ID", data=df_melted)

    # Titolo e layout
    plt.title('Performance Metrics per Client')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()

    # Salva il grafico
    plt.savefig(performance_dir + 'performance_metrics.png')
    #plt.show

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global currentRnd  # Dichiarazione esplicita della variabile globale

    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
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

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }

# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)

def server_fn(context: Context):
    server_config = ServerConfig(num_rounds=num_rounds)
    strategy = FedAvg(
        fraction_fit=1.0,  # Select all available clients
        fraction_evaluate=0.0,  # Disable evaluation
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )

app = ServerApp(server_fn=server_fn)
# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    # Start the server
    start_server(server_address="0.0.0.0:8080", config=ServerConfig(num_rounds=num_rounds))