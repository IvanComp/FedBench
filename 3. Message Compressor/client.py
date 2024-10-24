from flwr.client import ClientApp, NumPyClient
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Scalar,
    Context,
    Parameters,
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
from io import BytesIO
from taskA import (
    DEVICE as DEVICE_A,
    Net as NetA,
    get_weights as get_weights_A,
    load_data as load_data_A,
    set_weights as set_weights_A,
    train as train_A,
    test as test_A
)
import zlib
import pickle
import numpy as np
from APClient import ClientRegistry

MessageCompressorClientServer = True
MessageCompressorServerClient = True

CLIENT_ID = os.getenv("HOSTNAME")

# Instantiate a single instance of ClientRegistry for the client
client_registry = ClientRegistry()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, cid: str, model_type):
        self.cid = cid  
        self.model_type = "taskA"
        self.net = NetA().to(DEVICE_A)
        self.trainloader, self.testloader = load_data_A()  
        self.device = DEVICE_A

        client_registry.register_client(cid, model_type)

    def fit(self, parameters, config):
        cpu_start = psutil.cpu_percent(interval=None)
        compressed_parameters_hex = config.get("compressed_parameters_hex")
        numpy_arrays = None

        #DECOMPRESSION CODE Server to Client
        if MessageCompressorServerClient:
            compressed_parameters = bytes.fromhex(compressed_parameters_hex)
            decompressed_parameters = pickle.loads(zlib.decompress(compressed_parameters))
            numpy_arrays = [np.load(BytesIO(tensor)) for tensor in decompressed_parameters.tensors]
            numpy_arrays = [arr.astype(np.float32) for arr in numpy_arrays]
            parameters = numpy_arrays
            
        set_weights_A(self.net, parameters)
        results, training_time, start_comm_time = train_A(self.net, self.trainloader, self.testloader, epochs=1, device=self.device)       

        new_parameters = get_weights_A(self.net)
        compressed_parameters_hex = None

        #COMPRESSION CODE Client to Server
        if MessageCompressorClientServer:
            serialized_parameters = pickle.dumps(new_parameters)
            original_size = len(serialized_parameters)  
            compressed_parameters = zlib.compress(serialized_parameters)
            compressed_size = len(compressed_parameters)  
            compressed_parameters_hex = compressed_parameters.hex()
            reduction_bytes = original_size - compressed_size
            reduction_percentage = (reduction_bytes / original_size) * 100

            print(f"Compression from Client to Server: reduction of {reduction_bytes} bytes, {reduction_percentage:.2f}%")
        
        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        metrics = {
            "train_loss": results["train_loss"],
            "train_accuracy": results["train_accuracy"],
            "train_f1": results["train_f1"],
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
            "val_f1": results["val_f1"],
            "training_time": training_time,
            "cpu_usage": cpu_usage,
            "client_id": self.cid,
            "model_type": self.model_type,
            "start_comm_time": start_comm_time,
            "compressed_parameters_hex": compressed_parameters_hex,
        }

        if MessageCompressorClientServer:
            return [], len(self.trainloader.dataset), metrics
        else:
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

if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="server:8080",
        client=FlowerClient(cid=CLIENT_ID, model_type="taskA").to_client(),
    )