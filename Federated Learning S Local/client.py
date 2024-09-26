from flwr.client import ClientApp, NumPyClient
from flwr.common import Context  # Importa il tipo Context
import time
import csv
import os
import hashlib  # Importa hashlib per il hashing

from task import DEVICE, Net, get_weights, load_data, set_weights, train, test

# Crea la directory per i log delle performance
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Inizializza il file CSV, sovrascrivendolo
csv_file = os.path.join(performance_dir, 'performance.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)  # Sovrascrivi il file precedente

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time'])

# Carica modello e dati (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Definisci FlowerClient e client_fn
class FlowerClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting training...", flush=True)  # Log con il PID del client

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
        print(f"CLIENT {self.cid}: Training completed in {training_time:.2f} seconds", flush=True)

        # Logging del tempo di comunicazione
        print(f"CLIENT {self.cid}: Communication time: {communication_time:.2f} seconds", flush=True)

        total_time = training_time + communication_time

        # Append timing data to CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.cid, 0, training_time, communication_time, total_time])

        # Return weights, size of training data, and results
        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting evaluation...", flush=True)  # Log con il PID del client
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"CLIENT {self.cid}: Evaluation completed", flush=True)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    original_cid = context.node_id  # Flower dovrebbe fornire questo nel Context

    # Assicurati che original_cid sia una stringa
    original_cid_str = str(original_cid)

    # Utilizza un hash MD5 e tronca a 4 caratteri per ridurre la lunghezza di cid
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]  # Tronca a 4 caratteri

    return FlowerClient(cid=cid).to_client()

# Flower ClientApp usando client_fn
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    # Esempio di ID originale (puoi sostituirlo con il tuo metodo di generazione)
    original_cid = "1234567890"  # Sostituisci con il tuo metodo di generazione dell'ID originale

    # Assicurati che original_cid sia una stringa
    original_cid_str = str(original_cid)

    # Utilizza un hash MD5 e tronca a 4 caratteri
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]  # Tronca a 4 caratteri, ad esempio "1a2b"

    # Avvia il client Flower
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(cid=cid).to_client(),
    )