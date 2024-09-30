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
from APClient import ClientRegistry
import platform
import psutil
from datetime import datetime
import json

client_registry = ClientRegistry()

# Set the non-interactive backend of matplotlib
matplotlib.use('Agg')
# Get the absolute path of the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Global variable to keep track of the current round
currentRnd = 0
num_rounds = 2  # Total number of rounds

# Create the directory for performance logs
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Inizializza il file CSV, sovrascrivendo eventuali file esistenti
csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)'])

# Function to measure and log communication time
def measure_communication_time(start_time, end_time):
    communication_time = end_time - start_time
    print(f"Communication time: {communication_time:.2f} seconds")
    return communication_time


# Function to log the time of each round
def log_round_time(client_id, fl_round, training_time, communication_time, cpu_usage):  # Added CPU usage parameter
    total_time = training_time + communication_time
    print(
        f"CLIENT {client_id}: Round {fl_round} completed with total time {total_time:.2f} seconds and CPU usage {cpu_usage:.2f}%")

    # Save the data in the CSV
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [client_id, fl_round, training_time, communication_time, total_time, cpu_usage])  # Log CPU Usage

    # Update the client in the registry
    client_registry.update_client(client_id, True)

import seaborn as sns

# Function to generate performance graphs
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
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global currentRnd

    examples = [num_examples for num_examples, _ in metrics]

    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    # Raccogli le informazioni dei client dalle metriche e scrivi nel CSV
    for _, m in metrics:
        client_id = m.get("client_id")
        training_time = m.get("training_time")
        communication_time = m.get("communication_time")
        total_time = m.get("total_time")
        cpu_usage = m.get("cpu_usage")
        system_info_json = m.get("system_info")

        if client_id and system_info_json:
            # Deserializza system_info
            system_info = json.loads(system_info_json)

            client_registry.register_client(client_id, {})
            client_registry.registry[client_id]['system_info'] = system_info
            client_registry.registry[client_id]['last_update'] = datetime.now()

            # Scrivi le metriche nel CSV
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([client_id, currentRnd + 1, training_time, communication_time, total_time, cpu_usage])

    currentRnd += 1

    if currentRnd == num_rounds:
        generate_performance_graphs()
        generate_cpu_usage_graph()
        generate_total_time_graph()
        generate_training_time_graph()
        generate_communication_time_graph()
        client_registry.print_clients_info()

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
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    # Start the server
    start_server(server_address="0.0.0.0:8080", config=ServerConfig(num_rounds=num_rounds))
