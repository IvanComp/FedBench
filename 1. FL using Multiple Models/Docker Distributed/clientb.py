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
import hashlib
import psutil
import random
import torch
from taskB import (
    DEVICE as DEVICE_B,
    Net as NetB,
    get_weights as get_weights_B,
    load_data as load_data_B,
    set_weights as set_weights_B,
    train as train_B,
    test as test_B
)

from APClient import ClientRegistry

CLIENT_ID = os.getenv("HOSTNAME")

# Instantiate a single instance of ClientRegistry for the client
client_registry = ClientRegistry()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, cid: str, model_type):
        self.cid = cid 
        self.model_type = "taskB"
        self.net = NetB().to(DEVICE_B)
        self.trainloader, self.testloader = load_data_B()  
        self.device = DEVICE_B

        client_registry.register_client(cid, model_type)

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid} Successfully Configured. Target Model: {self.model_type}", flush=True)
        cpu_start = psutil.cpu_percent(interval=None)

        # Start training and record time
        training_start_time = time.time()
        set_weights_B(self.net, parameters)
        results, training_time = train_B(self.net, self.trainloader, self.testloader, epochs=1, device=self.device)
        training_end_time = time.time()
        
        training_time = training_end_time - training_start_time

        # Capture CPU usage
        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        communication_start_time = time.time()
        # Prepare the updated model weights and the metrics
        new_parameters = get_weights_B(self.net)

        metrics = {
            "train_loss": results["train_loss"],
            "train_accuracy": results["train_accuracy"],
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
            "training_time": training_time,
            "cpu_usage": cpu_usage,
            "client_id": self.cid,
            "model_type": self.model_type,
            "communication_start_time": communication_start_time,
        }

        return new_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid} ({self.model_type}): Starting evaluation.", flush=True)
        set_weights_B(self.net, parameters)
        loss, accuracy = test_B(self.net, self.testloader)

        print(f"CLIENT {self.cid} ({self.model_type}): Evaluation completed", flush=True)
        metrics = {
            "accuracy": accuracy,
            "client_id": self.cid,
            "model_type": self.model_type,
        }
        return loss, len(self.testloader.dataset), metrics

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    # Start the Flower client
    start_client(
        server_address="server:8080",
        client=FlowerClient(cid=CLIENT_ID,model_type="taskB").to_client(),
    )