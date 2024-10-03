from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from taskA import Net as NetA, get_weights as get_weights_A
from taskB import Net as NetB, get_weights as get_weights_B
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

global_metrics = {
    "taskA": {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []},
    "taskB": {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []},
}

# Set the non-interactive backend of matplotlib
matplotlib.use('Agg')
current_dir = os.path.abspath(os.path.dirname(__file__))

currentRnd = 0
num_rounds = 2

performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)','Task'])


def measure_communication_time(start_time, end_time):
    communication_time = end_time - start_time
    print(f"Communication time: {communication_time:.2f} seconds")
    return communication_time


def log_round_time(client_id, fl_round, training_time, communication_time, cpu_usage, model_type):
    total_time = training_time + communication_time
    print(
        f"CLIENT {client_id} ({model_type}): Round {fl_round} completed with total time {total_time:.2f} seconds and CPU usage {cpu_usage:.2f}%"
    )

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([client_id, fl_round, training_time, communication_time, total_time, cpu_usage, model_type])

    client_registry.update_client(client_id, True)

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


def weighted_average_global(metrics: List[Tuple[int, Metrics]], task_type: str) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    if total_examples == 0:
        return {
            "train_loss": float('inf'),
            "train_accuracy": 0.0,
        }

    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples

    # Memorizza le metriche nel dizionario globale
    global_metrics[task_type]["train_loss"].append(avg_train_loss)
    global_metrics[task_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[task_type]["val_loss"].append(avg_val_loss)
    global_metrics[task_type]["val_accuracy"].append(avg_val_accuracy)

    global currentRnd
    currentRnd += 1

    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type")  # Estrai model_type
        training_time = m.get("training_time")
        communication_time = m.get("communication_time")
        cpu_usage = m.get("cpu_usage")

        if client_id:
            if not client_registry.is_registered(client_id):
                client_registry.register_client(client_id, {})

            # Stampa model_type sul server
            print(f"Received model_type from Client {client_id}: {model_type}")

            # Log includendo model_type
            log_round_time(client_id, currentRnd, training_time, communication_time, cpu_usage, model_type)

    if currentRnd == num_rounds:
        generate_performance_graphs()
        generate_cpu_usage_graph()
        generate_total_time_graph()
        generate_training_time_graph()
        generate_communication_time_graph()
        client_registry.print_clients_info()
        print_final_results()

    return {
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
    }

# Inizializza i pesi separatamente per taskA e taskB
parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

def print_final_results():
    print("\nFinal results for taskA:")
    print(f"  Train loss: {global_metrics['taskA']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskA']['train_accuracy']}")
    print(f"  Val loss: {global_metrics['taskA']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskA']['val_accuracy']}")

    print("\nFinal results for taskB:")
    print(f"  Train loss: {global_metrics['taskB']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskB']['train_accuracy']}")
    print(f"  Val loss: {global_metrics['taskB']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskB']['val_accuracy']}")

def server_fn(context: Context):
    server_config = ServerConfig(num_rounds=num_rounds)
    # Determina dinamicamente il task_type basato sui client attivi
    active_clients = client_registry.get_active_clients()
    if active_clients:
        # Ottieni il tipo di task del primo client attivo
        first_client = next(iter(active_clients.values()))
        task_type = first_client['resources'].get("model_type", "taskA")
    else:
        # Valore di default se non ci sono client attivi
        task_type = "taskB"

    if task_type == "taskA":
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_available_clients=2,
            fit_metrics_aggregation_fn=lambda metrics: weighted_average_global(metrics, task_type="taskA"),
            initial_parameters=parametersA
        )
    elif task_type == "taskB":
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_available_clients=2,
            fit_metrics_aggregation_fn=lambda metrics: weighted_average_global(metrics, task_type="taskB"),
            initial_parameters=parametersB
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    from flwr.server import start_server

    start_server(server_address="0.0.0.0:8080", config=ServerConfig(num_rounds=num_rounds))