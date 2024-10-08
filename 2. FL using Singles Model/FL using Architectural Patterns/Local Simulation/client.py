from flwr.client import ClientApp, NumPyClient
from flwr.common import Context  # Import Context type
import time
from datetime import datetime
import csv
import os
import hashlib  # Import hashlib for hashing
import psutil  # CPU
import platform
import psutil
from datetime import datetime
import json


from task import DEVICE, Net, get_weights, load_data, set_weights, train, test

# Creazione della directory per i log di performance
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Inizializzazione del file CSV, sovrascrivendo eventuali file esistenti
csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if not os.path.exists(csv_file):  # Crea il file solo se non esiste
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)'])


def append_to_csv(data):
    """Scrivi i dati nel CSV evitando conflitti di accesso."""
    try:
        with open(csv_file, 'a', newline='') as file:
            # Blocca il file per impedire accessi concorrenti
            if os.name != 'nt':  # Su Windows, fcntl non è disponibile
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)

            writer = csv.writer(file)
            writer.writerow(data)

            if os.name != 'nt':
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    except PermissionError as e:
        print(f"PermissionError: {e}. Another process is using the file.")

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.resources = {}

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting training.", flush=True)
        cpu_start = psutil.cpu_percent(interval=None)

        comm_start_time = time.time()
        set_weights(net, parameters)
        results, training_time = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        comm_end_time = time.time()

        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        communication_time = comm_end_time - comm_start_time
        total_time = training_time + communication_time

        # Raccogli le informazioni di sistema del client
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_total": psutil.virtual_memory().total / (1024 ** 3),
            "ram_available": psutil.virtual_memory().available / (1024 ** 3),
            "python_version": platform.python_version(),
        }

        # Serializza system_info in una stringa JSON
        system_info_json = json.dumps(system_info)

        # Prepara le metriche da inviare al server
        metrics = {
            "train_loss": results["train_loss"],
            "train_accuracy": results["train_accuracy"],
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
            "training_time": training_time,
            "communication_time": communication_time,
            "total_time": total_time,
            "cpu_usage": cpu_usage,
            "client_id": self.cid,
            "system_info": system_info_json,  # Usa la stringa serializzata
        }

        return get_weights(net), len(trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting evaluation.", flush=True)
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"CLIENT {self.cid}: Evaluation completed", flush=True)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    original_cid = context.node_id
    original_cid_str = str(original_cid)

    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]

    return FlowerClient(cid=cid).to_client()

# Flower ClientApp using client_fn
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    original_cid = "1234567890"
    original_cid_str = str(original_cid)

    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(cid=cid).to_client(),
    )