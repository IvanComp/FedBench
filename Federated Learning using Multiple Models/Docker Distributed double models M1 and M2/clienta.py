from flwr.client import ClientApp, NumPyClient
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Scalar,
    Context,
)
from typing import Dict
import time
from datetime import datetime
import csv
import os
import json
import hashlib
import psutil
import random
import torch
from taskA import (
    DEVICE as DEVICE_A,
    Net as NetA,
    get_weights as get_weights_A,
    load_data as load_data_A,
    set_weights as set_weights_A,
    train as train_A,
    test as test_A
)

from APClient import ClientRegistry

CLIENT_ID = os.getenv("HOSTNAME")

# Istanzia un'unica istanza di ClientRegistry per il client
client_registry = ClientRegistry()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Creazione della directory per i log di performance
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

csv_file = os.path.join(performance_dir, '/FLwithAP_performance_metrics.csv')
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time', 'CPU Usage (%)', 'Task'])

class FlowerClient(NumPyClient):
    def __init__(self, cid, model_type):
        self.cid = CLIENT_ID
        self.model_type = model_type
        self.net = NetA().to(DEVICE_A)
        self.trainloader, self.testloader = load_data_A()  
        self.device = DEVICE_A

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid} Succesfully Configured. Target Model: {self.model_type}", flush=True)
        cpu_start = psutil.cpu_percent(interval=None)

        comm_start_time = time.time()
        set_weights_A(self.net, parameters)
        results, training_time = train_A(self.net, self.trainloader, self.testloader, epochs=1, device=self.device)
        new_parameters = get_weights_A(self.net)
        comm_end_time = time.time()
        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        communication_time = comm_end_time - comm_start_time
        total_time = training_time + communication_time

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
            "model_type": self.model_type,
        }

        return new_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid} ({self.model_type}): Starting evaluation.", flush=True)
        set_weights_A(self.net, parameters)
        loss, accuracy = test_A(self.net, self.testloader)

        print(f"CLIENT {self.cid} ({self.model_type}): Evaluation completed", flush=True)
        metrics = {
            "accuracy": accuracy,
            "client_id": self.cid,
            "model_type": self.model_type,
        }
        return loss, len(self.testloader.dataset), metrics

def client_fn(context: Context):
    original_cid = context.node_id
    original_cid_str = str(original_cid)  

    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]

    print(f"[DEBUG] Original CID: {original_cid_str}, Hashed CID: {cid}")
    
    print(f"[DEBUG] Creazione del client con original_cid: {original_cid_str}, assegnato cid: {cid}, model_type: {model_type}")
    client_registry.register_client(cid, model_type)  # Registra il client nel registro persistente

    return FlowerClient(cid=cid, model_type=model_type).to_client()

app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    original_cid_str = str(CLIENT_ID)
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]
    model_type = "taskA"

    # Start the Flower client
    start_client(
        server_address="server:8080",
        client=FlowerClient(cid=cid,model_type=model_type).to_client(),
    )