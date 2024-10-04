# Server.py

from typing import List, Tuple, Dict, Optional
from flwr.common import (
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    Context,
    FitIns,
    EvaluateIns,
    GetPropertiesIns,
    GetPropertiesRes,
)
from flwr.server import (
    ServerConfig,
    ServerApp,
    ServerAppComponents,
)
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log

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
import psutil

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
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)', 'Task'])


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

    # Store metrics in the global dictionary
    global_metrics[task_type]["train_loss"].append(avg_train_loss)
    global_metrics[task_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[task_type]["val_loss"].append(avg_val_loss)
    global_metrics[task_type]["val_accuracy"].append(avg_val_accuracy)

    global currentRnd
    currentRnd += 1

    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type")  # Extract model_type
        training_time = m.get("training_time")
        communication_time = m.get("communication_time")
        cpu_usage = m.get("cpu_usage")

        if client_id:
            if not client_registry.is_registered(client_id):
                client_registry.register_client(client_id, {})

            # Print model_type on the server
            print(f"Received model_type from Client {client_id}: {model_type}")

            # Log including model_type
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

# Initialize weights separately for taskA and taskB
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

# Definition of the custom strategy
class MultiModelStrategy(Strategy):
    def __init__(self, initial_parameters_a: Parameters, initial_parameters_b: Parameters):
        self.parameters_a = initial_parameters_a
        self.parameters_b = initial_parameters_b

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        # Return None because we use separate initial parameters for A and B
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Divide clients based on the model they are training
        clients = client_manager.sample(num_clients=client_manager.num_available())
        fit_configurations = []

        for client in clients:
            # Create GetPropertiesIns object
            get_properties_ins = GetPropertiesIns(config={})
            # Call get_properties with required arguments
            properties_res = client.get_properties(
                ins=get_properties_ins,
                timeout=None,
                # Provide group_id if required; use None or an empty string if not applicable
                group_id=None,
            )
            properties = properties_res.properties

            model_type = properties.get("model_type", "taskA")

            if model_type == "taskA":
                fit_ins = FitIns(self.parameters_a, {})
            elif model_type == "taskB":
                fit_ins = FitIns(self.parameters_b, {})
            else:
                continue  # Skip client if model_type is not recognized

            fit_configurations.append((client, fit_ins))

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
        ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
            # Separare i risultati per tipo di modello
            results_a = []
            results_b = []
            
            for client_proxy, fit_res in results:
                model_type = fit_res.metrics.get("model_type")
                if model_type == "taskA":
                    results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
                elif model_type == "taskB":
                    results_b.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            
            # Aggregare i parametri per taskA
            if results_a:
                self.parameters_a = self.aggregate_parameters(results_a, "taskA")
            
            # Aggregare i parametri per taskB
            if results_b:
                self.parameters_b = self.aggregate_parameters(results_b, "taskB")
            
            # Combinare le metriche aggregate
            metrics_aggregated = {
                "taskA": {
                    "train_loss": global_metrics["taskA"]["train_loss"][-1],
                    "train_accuracy": global_metrics["taskA"]["train_accuracy"][-1],
                    "val_loss": global_metrics["taskA"]["val_loss"][-1],
                    "val_accuracy": global_metrics["taskA"]["val_accuracy"][-1],
                },
                "taskB": {
                    "train_loss": global_metrics["taskB"]["train_loss"][-1],
                    "train_accuracy": global_metrics["taskB"]["train_accuracy"][-1],
                    "val_loss": global_metrics["taskB"]["val_loss"][-1],
                    "val_accuracy": global_metrics["taskB"]["val_accuracy"][-1],
                },
            }
            
            # Restituiamo i parametri per uno dei modelli per evitare l'errore
            # Se non vuoi aggiornare i parametri globali, puoi restituire i parametri attuali
            return self.parameters_a, metrics_aggregated

    def aggregate_parameters(self, results, task_type):
        # Aggregate weights using weighted average based on number of examples
        total_examples = sum([num_examples for _, num_examples, _ in results])
        new_weights = None

        metrics = []
        for client_params, num_examples, client_metrics in results:
            client_weights = parameters_to_ndarrays(client_params)
            weight = num_examples / total_examples
            if new_weights is None:
                new_weights = [w * weight for w in client_weights]
            else:
                new_weights = [nw + w * weight for nw, w in zip(new_weights, client_weights)]
            metrics.append((num_examples, client_metrics))

        # Aggregate metrics
        weighted_average_global(metrics, task_type)

        return ndarrays_to_parameters(new_weights)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Implement if necessary
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        # Implement if necessary
        return None

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Implement if necessary
        return None

def server_fn(context: Context):
    strategy = MultiModelStrategy(initial_parameters_a=parametersA, initial_parameters_b=parametersB)
    return ServerAppComponents(strategy=strategy)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    app.run(server_address="0.0.0.0:8080", config=ServerConfig(num_rounds=num_rounds))