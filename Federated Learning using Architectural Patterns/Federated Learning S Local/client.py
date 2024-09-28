from flwr.client import ClientApp, NumPyClient
from flwr.common import Context  # Import Context type
import time
import csv
import os
import hashlib  # Import hashlib for hashing

from task import DEVICE, Net, get_weights, load_data, set_weights, train, test

# Create the directory for performance logs
performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

# Initialize the CSV file, overwriting it
csv_file = os.path.join(performance_dir, 'performance.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)  # Overwrite the previous file

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time'])

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid

    def fit(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting training...", flush=True)  # Log with the client PID

        # Measure the initial communication time (receiving parameters from the server)
        comm_start_time = time.time()

        # Set weights and measure training time
        set_weights(net, parameters)
        results, training_time = train(net, trainloader, testloader, epochs=1, device=DEVICE)

        # Measure the final communication time (completion of the training cycle)
        comm_end_time = time.time()

        # Calculate the communication time
        communication_time = comm_end_time - comm_start_time

        # Log the training time
        print(f"CLIENT {self.cid}: Training completed in {training_time:.2f} seconds", flush=True)

        # Log the communication time
        print(f"CLIENT {self.cid}: Communication time: {communication_time:.2f} seconds", flush=True)

        total_time = training_time + communication_time

        # Append timing data to CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.cid, 0, training_time, communication_time, total_time])

        # Return weights, size of training data, and results
        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid}: Starting evaluation...", flush=True)  # Log with the client PID
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"CLIENT {self.cid}: Evaluation completed", flush=True)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    original_cid = context.node_id  # Flower should provide this in the Context

    # Ensure original_cid is a string
    original_cid_str = str(original_cid)

    # Use an MD5 hash and truncate to 4 characters to reduce the cid length
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]  # Truncate to 4 characters

    return FlowerClient(cid=cid).to_client()

# Flower ClientApp using client_fn
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    # Example of original ID (you can replace this with your own generation method)
    original_cid = "1234567890"  # Replace with your method of generating the original ID

    # Ensure original_cid is a string
    original_cid_str = str(original_cid)

    # Use an MD5 hash and truncate to 4 characters
    hash_object = hashlib.md5(original_cid_str.encode())
    cid = hash_object.hexdigest()[:4]  # Truncate to 4 characters, e.g., "1a2b"

    # Start the Flower client
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(cid=cid).to_client(),
    )
