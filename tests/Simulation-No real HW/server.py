import flwr as fl
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from concurrent import futures
import grpc

# Definizione della strategia personalizzata
class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.client_metrics = []
        self.round_metrics = []
        self.csv_path = os.path.join('performance', 'performance.csv')
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        with open(self.csv_path, 'w', newline='') as csvfile:
            fieldnames = ['round', 'client_id', 'training_time', 'communication_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        for client, fit_res in results:
            metrics = fit_res.metrics
            metric_entry = {
                'round': rnd,
                'client_id': client.cid,
                'training_time': metrics.get('training_time', 0),
                'communication_time': metrics.get('communication_time', 0)
            }
            self.client_metrics.append(metric_entry)

        self.save_metrics()

        return aggregated_parameters, aggregated_metrics

    def save_metrics(self):
        with open(self.csv_path, 'a', newline='') as csvfile:
            fieldnames = ['round', 'client_id', 'training_time', 'communication_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for entry in self.client_metrics:
                writer.writerow(entry)
        self.client_metrics = []

def main():
    strategy = SaveMetricsStrategy()

    # Limita i thread di gRPC
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3, round_timeout=120),  # Timeout ridotto
        strategy=strategy
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Errore nell'avvio del server: {e}")
