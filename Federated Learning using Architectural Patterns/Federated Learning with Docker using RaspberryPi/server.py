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

# Set the non-interactive backend of matplotlib
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
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time'])

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

def generate_performance_graphs():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Check that the CSV file exists
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"The CSV file '{csv_file}' does not exist.")

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Check that the required columns exist
    required_columns = ['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"The CSV file must contain the columns: {required_columns}")

    # Map client IDs to "client 1", "client 2", etc.
    unique_clients = df['Client ID'].unique()
    client_mapping = {original_id: f"client {i + 1}" for i, original_id in enumerate(unique_clients)}

    # Debug: print the created mapping
    print("Client ID mapping:")
    for original, mapped in client_mapping.items():
        print(f"{original} -> {mapped}")

    # Apply the mapping to the DataFrame
    df['Client ID'] = df['Client ID'].map(client_mapping)

    # Overwrite the 'FL Round' column with incremental values starting from 1
    num_clients = len(unique_clients)
    df = df.reset_index(drop=True)  # Ensure the index is sequential
    df['FL Round'] = (df.index // num_clients) + 1

    df[['Training Time', 'Communication Time', 'Total Time']] = df[
        ['Training Time', 'Communication Time', 'Total Time']].round(2)

    # Write the changes to the CSV file
    df.to_csv(csv_file, index=False)
    print(f"CSV file updated and saved at '{csv_file}'.")

    plt.figure(figsize=(12, 6))

    # Create histograms for Training Time, Communication Time, and Total Time
    df_melted = df.melt(
        id_vars=["Client ID"],
        value_vars=["Training Time", "Communication Time", "Total Time"],
        var_name="Metric",
        value_name="Time (seconds)"
    )

    sns.barplot(x="Metric", y="Time (seconds)", hue="Client ID", data=df_melted)

    # Title and layout
    plt.title('Performance Metrics per Client')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Metric')
    plt.legend(title='Client ID')
    plt.tight_layout()

    # Save the graph
    graph_path = os.path.join(performance_dir, 'performance_metrics.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved at '{graph_path}'.")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global currentRnd  # Explicit declaration of the global variable

    examples = [num_examples for num_examples, _ in metrics]

    # Multiply the accuracy of each client by the number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    currentRnd += 1

    # If we are in the last round, generate the graphs
    if currentRnd == num_rounds:
        generate_performance_graphs()

    # Aggregate and return custom metrics (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
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
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Start the server with the custom strategy
    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
