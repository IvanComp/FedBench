from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from task import Net, get_weights
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# Imposta il backend non interattivo di matplotlib
matplotlib.use('Agg')

# Variabile globale per tenere traccia del round corrente
currentRnd = 0
num_rounds = 3  # Numero totale di round

# Ottieni il percorso assoluto della directory corrente
current_dir = os.path.abspath(os.path.dirname(__file__))

# Crea la directory per i log delle performance
performance_dir = os.path.join(current_dir, 'performance')
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Definisci il percorso del file CSV
csv_file = os.path.join(performance_dir, 'performance.csv')

# Inizializza il file CSV, sovrascrivendolo
if os.path.exists(csv_file):
    try:
        os.remove(csv_file)
        print(f"File '{csv_file}' rimosso con successo.")
    except OSError as e:
        print(f"Errore nella rimozione del file: {e}")

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time'])

# Funzione per misurare e loggare il tempo di comunicazione
def measure_communication_time(start_time, end_time):
    communication_time = end_time - start_time
    print(f"Communication time: {communication_time:.2f} seconds")
    return communication_time

# Funzione per loggare il tempo di ogni round
def log_round_time(client_id, fl_round, training_time, communication_time):
    total_time = training_time + communication_time
    print(f"CLIENT {client_id}: Round {fl_round} completed with total time {total_time:.2f} seconds")

    # Salva i dati nel CSV
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([client_id, fl_round, training_time, communication_time, total_time])

def generate_performance_graphs():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Verifica che il file CSV esista
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Il file CSV '{csv_file}' non esiste.")

    # Leggi il file CSV
    df = pd.read_csv(csv_file)

    # Verifica che le colonne necessarie esistano
    required_columns = ['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"Il file CSV deve contenere le colonne: {required_columns}")

    # Mappa gli ID dei client a "client 1", "client 2", ecc.
    unique_clients = df['Client ID'].unique()
    client_mapping = {original_id: f"client {i + 1}" for i, original_id in enumerate(unique_clients)}

    # Debug: stampa la mappatura creata
    print("Mappatura degli ID dei client:")
    for original, mapped in client_mapping.items():
        print(f"{original} -> {mapped}")

    # Applica la mappatura al DataFrame
    df['Client ID'] = df['Client ID'].map(client_mapping)

    # Sovrascrivi la colonna 'FL Round' con valori incrementali a partire da 1
    num_clients = len(unique_clients)
    df = df.reset_index(drop=True)  # Assicurati che l'indice sia sequenziale
    df['FL Round'] = (df.index // num_clients) + 1

    df[['Training Time', 'Communication Time', 'Total Time']] = df[
        ['Training Time', 'Communication Time', 'Total Time']].round(2)

    # Scrivi le modifiche sul file CSV
    df.to_csv(csv_file, index=False)
    print(f"File CSV aggiornato e salvato in '{csv_file}'.")

    plt.figure(figsize=(12, 6))

    # Crea gli istogrammi per Training Time, Communication Time e Total Time
    df_melted = df.melt(
        id_vars=["Client ID"],
        value_vars=["Training Time", "Communication Time", "Total Time"],
        var_name="Metric",
        value_name="Time (seconds)"
    )

    sns.barplot(x="Metric", y="Time (seconds)", hue="Client ID", data=df_melted)

    # Titolo e layout
    plt.title('Performance Metrics per Client')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Metric')
    plt.legend(title='Client ID')
    plt.tight_layout()

    # Salva il grafico
    graph_path = os.path.join(performance_dir, 'performance_metrics.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Grafico salvato in '{graph_path}'.")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global currentRnd  # Dichiarazione esplicita della variabile globale

    examples = [num_examples for num_examples, _ in metrics]

    # Moltiplica la precisione di ogni client per il numero di esempi usati
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    currentRnd += 1

    # Se siamo nell'ultimo round, genera i grafici
    if currentRnd == num_rounds:
        generate_performance_graphs()

    # Aggrega e ritorna le metriche personalizzate (media ponderata)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }

# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)

def server_fn(context: Context):
    server_config = ServerConfig(num_rounds=num_rounds)
    strategy = FedAvg(
        fraction_fit=1.0,  # Seleziona tutti i client disponibili
        fraction_evaluate=0.0,  # Disabilita la valutazione
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )

app = ServerApp(server_fn=server_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    # Start the server
    start_server(server_address="flwr_server:8080", config=ServerConfig(num_rounds=num_rounds))