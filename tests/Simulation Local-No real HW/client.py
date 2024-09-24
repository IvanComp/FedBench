from flwr.client import ClientApp, NumPyClient
from flwr.common import Context  # Importa il tipo Context
import time
import csv
import os

from task import DEVICE, Net, get_weights, load_data, set_weights, train, test

# Crea la directory per i log delle performance
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Inizializza il file CSV, sovrascrivendolo
csv_file = performance_dir + 'performance.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)  # Sovrascrivi il file precedente

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'Training Time', 'Communication Time', 'Total Time'])

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self):
        self.client_id = os.getpid()  # Usa il PID come identificativo del client

    def fit(self, parameters, config):
        print(f"CLIENT {self.client_id}: Starting training...", flush=True)  # Log con il PID del client

        # Misura il tempo di comunicazione iniziale (ricezione dei parametri dal server)
        comm_start_time = time.time()

        # Set weights e misura il tempo di training
        set_weights(net, parameters)
        results, training_time = train(net, trainloader, testloader, epochs=1, device=DEVICE)

        # Misura il tempo di comunicazione finale (completamento del ciclo di allenamento)
        comm_end_time = time.time()

        # Calcola il communication time
        communication_time = comm_end_time - comm_start_time

        # Logging del tempo di training
        print(f"CLIENT {self.client_id}: Training completed in {training_time:.2f} seconds", flush=True)

        # Logging del tempo di comunicazione
        print(f"CLIENT {self.client_id}: Communication time: {communication_time:.2f} seconds", flush=True)

        total_time = training_time + communication_time

        # Append timing data to CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.client_id, training_time, communication_time, total_time])

        # Return weights and size
        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.client_id}: Starting evaluation...", flush=True)  # Log con il PID del client
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"CLIENT {self.client_id}: Evaluation completed", flush=True)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Crea e restituisci un'istanza di Flower Client."""
    return FlowerClient().to_client()


# Flower ClientApp usando client_fn
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
