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
from APClient import ClientRegistry
import platform
import psutil
from datetime import datetime
import json

client_registry = ClientRegistry()

matplotlib.use('Agg')

# Device configuration
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

# Function to get model weights
def get_weights(net):
    # Ensure consistent key ordering
    state_dict = net.state_dict()
    ordered_state_dict = OrderedDict(sorted(state_dict.items()))
    return [val.cpu().numpy() for key, val in ordered_state_dict.items()]

# Global variable to keep track of the current round
currentRnd = 0
num_rounds = 3  # Total number of rounds

# Get the absolute path of the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Create the directory for performance logs
performance_dir = os.path.join(current_dir, 'performance')
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Define the path of the CSV file
csv_file = os.path.join(performance_dir, 'performance.csv')

# Initialize the CSV file, overwriting it if it exists
if os.path.exists(csv_file):
    try:
        os.remove(csv_file)
        print(f"File '{csv_file}' successfully removed.")
    except OSError as e:
        print(f"Error removing the file: {e}")

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time','CPU Usage (%)'])

# Function to measure and log communication time
def measure_communication_time(start_time, end_time):
    communication_time = end_time - start_time
    print(f"Communication time: {communication_time:.2f} seconds")
    return communication_time

# Function to log the time of each round
def log_round_time(client_id, fl_round, training_time, communication_time):
    total_time = training_time + communication_time
    print(f"CLIENT {client_id}: Round {fl_round} completed with total time {total_time:.2f} seconds")

    # Save the data in the CSV
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([client_id, fl_round, training_time, communication_time, total_time])

    client_registry.update_client(client_id, True)

def generate_performance_graphs():
    sns.set_theme(style="ticks")

    df = pd.read_csv(csv_file)

    unique_clients = df['Client ID'].unique()
    client_mapping = {original_id: f"client {i + 1}" for i, original_id in enumerate(unique_clients)}
    df['Client ID'] = df['Client ID'].map(client_mapping)

    num_clients = len(unique_clients)
    df = df.reset_index(drop=True)
    df['FL Round'] = (df.index // num_clients) + 1
    df[['Training Time', 'Communication Time', 'Total Time']] = df[['Training Time', 'Communication Time', 'Total Time']].round(2)
    df.to_csv(csv_file, index=False)

    plt.figure(figsize=(12, 6))
    df_melted = df.melt(id_vars=["Client ID"], value_vars=["Training Time", "Communication Time", "Total Time"],
                        var_name="Metric", value_name="Time (seconds)")
    sns.barplot(x="Metric", y="Time (seconds)", hue="Client ID", data=df_melted)
    plt.title('Performance Metrics per Client', fontweight='bold')
    plt.ylabel('Time (seconds)', fontweight='bold')
    plt.legend(title='Client ID', title_fontsize='13', fontsize='10', loc='best', frameon=True)
    plt.tight_layout()

    graph_path = os.path.join(performance_dir, 'performance_metrics.pdf')
    plt.savefig(graph_path, format="pdf")
    plt.close()

# Function to generate CPU usage graph
def generate_cpu_usage_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Client ID", y="CPU Usage (%)", data=df)
    plt.title('CPU Usage per Client', fontweight='bold')
    plt.ylabel('CPU Usage (%)', fontweight='bold')
    plt.xlabel('Client ID', fontweight='bold')
    plt.tight_layout()

    cpu_graph_path = os.path.join(performance_dir, 'cpu_usage_per_client.pdf')
    plt.savefig(cpu_graph_path, format="pdf")
    plt.close()

# Function to generate total time graph
def generate_total_time_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='FL Round', y='Total Time', hue='Client ID', data=df, marker="o", markersize=8)
    plt.title('Total Time per Round per Client', fontweight='bold')
    plt.ylabel('Total Time (seconds)', fontweight='bold')
    plt.xlabel('FL Round', fontweight='bold')
    plt.tight_layout()

    line_graph_path = os.path.join(performance_dir, 'totalTime_round.pdf')
    plt.savefig(line_graph_path, format="pdf")
    plt.close()

# Function to generate training time graph
def generate_training_time_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='FL Round', y='Training Time', hue='Client ID', data=df, marker="o", markersize=8)
    plt.title('Training Time per Round per Client', fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontweight='bold')
    plt.xlabel('FL Round', fontweight='bold')
    plt.tight_layout()

    line_graph_path = os.path.join(performance_dir, 'trainingTime_round.pdf')
    plt.savefig(line_graph_path, format="pdf")
    plt.close()

# Function to generate communication time graph
def generate_communication_time_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='FL Round', y='Communication Time', hue='Client ID', data=df, marker="o", markersize=8)
    plt.title('Communication Time per Round per Client', fontweight='bold')
    plt.ylabel('Communication Time (seconds)', fontweight='bold')
    plt.xlabel('FL Round', fontweight='bold')
    plt.tight_layout()

    line_graph_path = os.path.join(performance_dir, 'communicationTime_round.pdf')
    plt.savefig(line_graph_path, format="pdf")
    plt.close()

# Define metric aggregation function
def weighted_average_global(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    if total_examples == 0:
        print("No examples available, skipping aggregation.")
        return {
            "train_loss": float('inf'),
            "train_accuracy": 0.0,
        }

    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    global currentRnd
    currentRnd += 1

    # Aggiungi la registrazione e l'aggiornamento dei client nel registro
    for _, m in metrics:
        client_id = m.get("client_id")
        training_time = m.get("training_time")
        communication_time = m.get("communication_time")
        cpu_usage = m.get("cpu_usage")
        if client_id:
            # Registra o aggiorna il client nel registry
            if not client_registry.is_registered(client_id):
                # Registra il client se non è ancora registrato
                client_registry.register_client(client_id, {})
            # Aggiorna il client come attivo
            client_registry.update_client(client_id, True)
            # Log delle metriche del round
            log_round_time(client_id, currentRnd, training_time, communication_time, cpu_usage)

    # Genera i grafici e stampa le informazioni dei client alla fine dell'ultimo round
    if currentRnd == num_rounds:
        generate_performance_graphs()
        generate_cpu_usage_graph()
        generate_total_time_graph()
        generate_training_time_graph()
        generate_communication_time_graph()
        client_registry.print_clients_info()

    return {
        "train_loss": sum(train_losses) / total_examples,
        "train_accuracy": sum(train_accuracies) / total_examples,
        "val_loss": sum(val_losses) / total_examples,
        "val_accuracy": sum(val_accuracies) / total_examples,
    }


# Initialize model parameters
net = Net()
ndarrays = get_weights(net)
parameters = ndarrays_to_parameters(ndarrays)

if __name__ == "__main__":
    from flwr.server import start_server

    # Define the custom strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Select all available clients
        fraction_evaluate=0.0,  # Disable evaluation
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average_global,
        initial_parameters=parameters,
    )

    # Start the server with the custom strategy
    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
