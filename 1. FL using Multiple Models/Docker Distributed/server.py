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
)
from flwr.server import (
    ServerConfig,
    ServerApp,
    ServerAppComponents,
    start_server
)
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from prometheus_client import Gauge, start_http_server
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

num_rounds = int(os.getenv("NUM_ROUNDS", 2))
currentRnd = 0


performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)', 'Task',
                     'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'SRT1', 'SRT2'])

def measure_communication_time(start_time, end_time):
    communication_time = end_time - start_time
    print(f"Communication time: {communication_time:.2f} seconds")
    return communication_time

def log_round_time(client_id, fl_round, training_time, communication_time, total_time, cpu_usage, model_type, already_logged, srt1, srt2):
    total_time = training_time + communication_time
    
    train_loss = round(global_metrics[model_type]["train_loss"][-1]) if global_metrics[model_type]["train_loss"] else 'N/A'
    train_accuracy = round(global_metrics[model_type]["train_accuracy"][-1], 2) if global_metrics[model_type]["train_accuracy"] else 'N/A'
    val_loss = round(global_metrics[model_type]["val_loss"][-1]) if global_metrics[model_type]["val_loss"] else 'N/A'
    val_accuracy = round(global_metrics[model_type]["val_accuracy"][-1], 2) if global_metrics[model_type]["val_accuracy"] else 'N/A'

    if already_logged:
        train_loss = ""
        train_accuracy = ""
        val_loss = ""
        val_accuracy = ""
        srt1 = ""
        srt2 = ""

    srt1_rounded = round(srt1) if isinstance(srt1, (int, float)) else srt1
    srt2_rounded = round(srt2) if isinstance(srt2, (int, float)) else srt2

    print(
        f"CLIENT {client_id} ({model_type}): Round {fl_round+1} completed with: Training Time {round(training_time)} seconds, "
        f"Communication Time {round(communication_time)} seconds, Total Time {round(total_time)} seconds, CPU usage {round(cpu_usage)}%"
    )

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([client_id, fl_round+1, round(training_time), round(communication_time), round(total_time), round(cpu_usage), model_type,
                         train_loss, train_accuracy, val_loss, val_accuracy, srt1_rounded, srt2_rounded])

    client_registry.update_client(client_id, True)

# Function to generate performance graphs
def generate_performance_graphs():
    sns.set_theme(style="ticks")

    df = pd.read_csv(csv_file)
    print(df.head())  # Debugging: check CSV content

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
    print(f"Saving the graph to: {graph_path}")  # Debugging: check save path
    plt.savefig(graph_path, format="pdf")
    
    # Check if the file was created
    if os.path.exists(graph_path):
        print(f"The graph was successfully created in {graph_path}")
    else:
        print(f"Error: the file {graph_path} was not created.")
        
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
    print(f"Saving the graph to: {cpu_graph_path}")
    plt.savefig(cpu_graph_path, format="pdf")
    
    if os.path.exists(cpu_graph_path):
        print(f"The graph was successfully created in {cpu_graph_path}")
    else:
        print(f"Error: the file {cpu_graph_path} was not created.")
        
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
    print(f"Saving the graph to: {line_graph_path}")
    plt.savefig(line_graph_path, format="pdf")
    
    if os.path.exists(line_graph_path):
        print(f"The graph was successfully created in {line_graph_path}")
    else:
        print(f"Error: the file {line_graph_path} was not created.")
        
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
    print(f"Saving the graph to: {line_graph_path}")
    plt.savefig(line_graph_path, format="pdf")
    
    if os.path.exists(line_graph_path):
        print(f"The graph was successfully created in {line_graph_path}")
    else:
        print(f"Error: the file {line_graph_path} was not created.")
        
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
    print(f"Saving the graph to: {line_graph_path}")
    plt.savefig(line_graph_path, format="pdf")
    
    if os.path.exists(line_graph_path):
        print(f"The graph was successfully created in {line_graph_path}")
    else:
        print(f"Error: the file {line_graph_path} was not created.")
        
    plt.close()


def weighted_average_global(metrics: List[Tuple[int, Metrics]], task_type: str, srt1, srt2, communication_time) -> Metrics:
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

    already_logged = False

    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type")
        training_time = m.get("training_time")
        cpu_usage = m.get("cpu_usage")
       
        if client_id:
            if not client_registry.is_registered(client_id):
                client_registry.register_client(client_id, model_type)

            total_time = training_time + communication_time
            # Log including model_type, srt1, and srt2
            log_round_time(client_id, currentRnd, training_time, communication_time, total_time, cpu_usage, model_type, already_logged, srt1, srt2)

            already_logged = True

    return {
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
    }

# Initialize weights separately for taskA and taskB
parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

def print_results():
    # Collect clients for each task using cid
    clients_taskA = [cid for cid, model in client_model_mapping.items() if model == "taskA" and len(cid) <= 12]
    clients_taskB = [cid for cid, model in client_model_mapping.items() if model == "taskB" and len(cid) <= 12]

    print("\nResults for Model A:")
    print(f"  Clients: {clients_taskA}")
    print(f"  Train loss: {global_metrics['taskA']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskA']['train_accuracy']}")
    print(f"  Val loss: {global_metrics['taskA']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskA']['val_accuracy']}")

    print("\nResults for Model B:")
    print(f"  Clients: {clients_taskB}")
    print(f"  Train loss: {global_metrics['taskB']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskB']['train_accuracy']}")
    print(f"  Val loss: {global_metrics['taskB']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskB']['val_accuracy']}\n")

# Initialize the client_model_mapping dictionary
client_model_mapping = {}

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
        min_clients = 2

        # Wait until there are enough clients
        while client_manager.num_available() < min_clients:
            time.sleep(1)  

        # Sample available clients after reaching the minimum number
        clients = client_manager.sample(num_clients=min_clients)

        fit_configurations = []

        task_flag = True  # Start with True for taskA

        for i, client in enumerate(clients):
            client_id = client.cid

            # Alternate between taskA and taskB
            if task_flag:
                fit_ins = FitIns(self.parameters_a, {})
                model_type = "taskA"
            else:
                fit_ins = FitIns(self.parameters_b, {})
                model_type = "taskB"
            
            # Alternate the flag
            task_flag = not task_flag
            
            # Map the model to the client
            client_model_mapping[client_id] = model_type

            # Add the configuration
            fit_configurations.append((client, fit_ins))
        
        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
        ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:

        results_a = []
        results_b = []
        training_start_time = None
        training_end_time = None

        # Inizializziamo srt1 e srt2 a 'N/A' per garantire che abbiano sempre un valore
        srt1 = 'N/A'
        srt2 = 'N/A'
        
        global aggregation_end_time

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id")
            model_type = fit_res.metrics.get("model_type")
            client_training_start_time = fit_res.metrics.get("training_start_time")
            client_training_end_time = fit_res.metrics.get("training_end_time")
            communication_start_time = fit_res.metrics.get("communication_start_time")
            communication_time = time.time() - communication_start_time

            if training_start_time is None or client_training_start_time < training_start_time:
                training_start_time = client_training_start_time

            if training_end_time is None or client_training_end_time > training_end_time:
                training_end_time = client_training_end_time

            # Calcoliamo srt1 solo se entrambi i tempi sono disponibili
            if training_start_time and training_end_time:
                srt1 = round(training_end_time - training_start_time)

            client_model_mapping[client_id] = model_type
            print(f"[DEBUG] Results received from client {client_id}, Model Type: {model_type}")

            if model_type == "taskA":
                results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            elif model_type == "taskB":
                results_b.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            else:
                print(f"[DEBUG] Unknown Model Type for client {client_id}")
                continue

        # Aggrega i parametri per taskA
        if results_a:
            print(f"[DEBUG] Aggregating parameters for taskA with {len(results_a)} results")
            self.parameters_a = self.aggregate_parameters(results_a, "taskA", srt1, srt2, communication_time)

        # Aggrega i parametri per taskB
        if results_b:
            print(f"[DEBUG] Aggregating parameters for taskB with {len(results_b)} results")
            self.parameters_b = self.aggregate_parameters(results_b, "taskB", srt1, srt2, communication_time)

        # Combiniamo le metriche aggregate
        metrics_aggregated = {
            "taskA": {
                "train_loss": global_metrics["taskA"]["train_loss"][-1] if global_metrics["taskA"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskA"]["train_accuracy"][-1] if global_metrics["taskA"]["train_accuracy"] else None,
                "val_loss": global_metrics["taskA"]["val_loss"][-1] if global_metrics["taskA"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskA"]["val_accuracy"][-1] if global_metrics["taskA"]["val_accuracy"] else None,
            },
            "taskB": {
                "train_loss": global_metrics["taskB"]["train_loss"][-1] if global_metrics["taskB"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskB"]["train_accuracy"][-1] if global_metrics["taskB"]["train_accuracy"] else None,
                "val_loss": global_metrics["taskB"]["val_loss"][-1] if global_metrics["taskB"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskB"]["val_accuracy"][-1] if global_metrics["taskB"]["val_accuracy"] else None,
            },
        }

        print_results()

        # Calcoliamo srt2 dopo aver completato l'aggregazione
        aggregation_end_time = time.time()
        if training_start_time:
            srt2 = round(aggregation_end_time - training_start_time)

        global currentRnd
        currentRnd += 1

        if currentRnd == num_rounds:
            print("Starting graph generation...")
            generate_performance_graphs()
            generate_cpu_usage_graph()
            generate_total_time_graph()
            generate_training_time_graph()
            generate_communication_time_graph()

        return (self.parameters_a, self.parameters_b), metrics_aggregated

    def aggregate_parameters(self, results, task_type, srt1, srt2,communication_time):
        # Aggregate weights using weighted average based on number of examples
        total_examples = sum([num_examples for _, num_examples, _ in results])
        new_weights = None

        metrics = []
        for client_params, num_examples, client_metrics in results:
            client_weights = parameters_to_ndarrays(client_params)
            weight = num_examples / total_examples
            print(f"[DEBUG] Aggregating parameters for {task_type}, num_examples: {num_examples}, weight: {weight}")
            if new_weights is None:
                new_weights = [w * weight for w in client_weights]
            else:
                new_weights = [nw + w * weight for nw, w in zip(new_weights, client_weights)]
            metrics.append((num_examples, client_metrics))

        # Aggregate metrics
        weighted_average_global(metrics, task_type, srt1, srt2, communication_time)

        return ndarrays_to_parameters(new_weights)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        return None

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

if __name__ == "__main__":
    # Start Prometheus Metrics Server
    start_http_server(8000)
    
    strategy = MultiModelStrategy(
        initial_parameters_a=parametersA,  
        initial_parameters_b=parametersB,  
    )

    start_server(
        server_address="[::]:8080",  
        config=ServerConfig(num_rounds=num_rounds),  
        strategy=strategy, 
    )