# server.py
import flwr as fl
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Importazione aggiuntiva

from typing import List, Tuple, Dict, Any

# Definizione della strategia personalizzata
class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.client_metrics = []
        self.round_metrics = []
        # Percorso del file CSV
        self.csv_path = os.path.join('performance', 'performance.csv')
        # Assicurati che il file non esista già
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        # Crea il file CSV e scrivi l'intestazione
        with open(self.csv_path, 'w', newline='') as csvfile:
            fieldnames = ['round', 'client_id', 'training_time', 'communication_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, BaseException]],
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:  # Modifica qui
        # Chiama la funzione originale di aggregazione
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # Raccolta delle metriche dai client
        for client, fit_res in results:
            metrics = fit_res.metrics
            metric_entry = {
                'round': rnd,
                'client_id': client.cid,
                'training_time': metrics.get('training_time', 0),
                'communication_time': metrics.get('communication_time', 0)
            }
            self.client_metrics.append(metric_entry)

        # Salva le metriche nel file CSV
        self.save_metrics()

        return aggregated_parameters, aggregated_metrics

    def save_metrics(self):
        with open(self.csv_path, 'a', newline='') as csvfile:
            fieldnames = ['round', 'client_id', 'training_time', 'communication_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for entry in self.client_metrics:
                writer.writerow(entry)
        # Svuota la lista dopo aver salvato
        self.client_metrics = []

def generate_reports(csv_path: str):
    # Imposta lo stile dei grafici
    sns.set(style="whitegrid")

    # Caricamento dei dati
    data = pd.read_csv(csv_path)

    # Verifica dei dati caricati
    print("Esempio dei dati raccolti:")
    print(data.head())

    # Grafico del tempo di training per client
    plt.figure(figsize=(12, 6))
    sns.barplot(x='client_id', y='training_time', data=data, palette="Blues_d")
    plt.xlabel('ID Client')
    plt.ylabel('Tempo di Training (s)')
    plt.title('Tempo di Training per Client')
    plt.savefig(os.path.join('performance', 'training_time_per_client.png'))
    plt.close()

    # Grafico del tempo di comunicazione per client
    plt.figure(figsize=(12, 6))
    sns.barplot(x='client_id', y='communication_time', data=data, palette="Greens_d")
    plt.xlabel('ID Client')
    plt.ylabel('Tempo di Comunicazione (s)')
    plt.title('Tempo di Comunicazione per Client')
    plt.savefig(os.path.join('performance', 'communication_time_per_client.png'))
    plt.close()

    # Grafico dei tempi di training per round per client
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='round', y='training_time', hue='client_id', data=data, marker="o")
    plt.xlabel('Round')
    plt.ylabel('Tempo di Training (s)')
    plt.title('Tempo di Training per Client per Round')
    plt.legend(title='ID Client', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join('performance', 'training_time_per_client_round.png'))
    plt.close()

    # Grafico dei tempi di comunicazione per round per client
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='round', y='communication_time', hue='client_id', data=data, marker="o")
    plt.xlabel('Round')
    plt.ylabel('Tempo di Comunicazione (s)')
    plt.title('Tempo di Comunicazione per Client per Round')
    plt.legend(title='ID Client', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join('performance', 'communication_time_per_client_round.png'))
    plt.close()

    print("Report generati nella cartella 'performance/'.")

def main():
    # Configura la strategia personalizzata
    strategy = SaveMetricsStrategy()

    # Avvia il server Flower
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

    # Dopo che il server ha terminato, genera i report
    csv_path = os.path.join('performance', 'performance.csv')
    if os.path.exists(csv_path):
        generate_reports(csv_path)
    else:
        print(f"Il file {csv_path} non è stato trovato. Verifica che il federated learning abbia raccolto le metriche.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Errore nell'avvio del server: {e}")
