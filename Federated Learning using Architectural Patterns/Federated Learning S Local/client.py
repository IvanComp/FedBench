from flwr.client import ClientApp, NumPyClient
import psutil  # To monitor system resources
import time
import csv
import os
import hashlib
from APClient import ClientRegistry  # Import the client registry
from task import DEVICE, Net, get_weights, load_data, set_weights, train, test

# Set up the performance log file
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)'])

# Initialize the client registry
client_registry = ClientRegistry()

# Load model and data
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define FlowerClient
class FlowerClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        # If the client is already registered, load its resources from the registry
        if client_registry.is_client_registered(self.cid):
            self.resources = client_registry.get_client_resources(self.cid)
            print(f"CLIENT {self.cid}: Resources loaded from registry.")
        else:
            # If not registered, gather the resources and register the client
            self.resources = self.get_resources()
            client_registry.register_client(self.cid, self.resources)

    def get_resources(self):
        """Collect current resource information from the client."""
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        return {
            "cpu": cpu_usage,
            "memory_available": memory_info.available,
        }

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting training.", flush=True)

        # Start CPU and communication time monitoring
        cpu_start = psutil.cpu_percent(interval=None)
        comm_start_time = time.time()

        set_weights(net, parameters)
        results, training_time = train(net, trainloader, testloader, epochs=1, device=DEVICE)

        comm_end_time = time.time()
        cpu_end = psutil.cpu_percent(interval=None)

        # Log performance data
        cpu_usage = (cpu_start + cpu_end) / 2
        communication_time = comm_end_time - comm_start_time
        total_time = training_time + communication_time

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.cid, 0, training_time, communication_time, total_time, cpu_usage])

        # Update client resources in the registry
        client_registry.update_client(self.cid, {
            "cpu": cpu_usage,
            "communication_time": communication_time,
        })

        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting evaluation.", flush=True)
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"CLIENT {self.cid}: Evaluation completed", flush=True)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

    def on_disconnect(self):
        client_registry.remove_client(self.cid)

def client_fn(context):
    original_cid = context.node_id
    cid = hashlib.md5(str(original_cid).encode()).hexdigest()[:4]
    return FlowerClient(cid=cid).to_client()

# Flower ClientApp using client_fn
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    original_cid = "1234567890"  # Example of original client ID
    cid = hashlib.md5(str(original_cid).encode()).hexdigest()[:4]

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(cid=cid).to_client(),
    )
