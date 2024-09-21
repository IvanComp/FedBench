# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definizione del modello
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Caricamento dei dati
def load_data(test=False):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    if test:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=2
    )
    return loader

# Funzione di training
def train(net, trainloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        logger.info(f"Inizio epoch {epoch+1}")
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    logger.info("Training completato.")

# Funzione di valutazione
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    logger.info(f"Test completato. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy

# Implementazione del client Flower
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = Net()
        self.trainloader = load_data(test=False)
        self.testloader = load_data(test=True)
        logger.info("Client inizializzato con successo.")

    def get_parameters(self, config):
        logger.info("Richiesta dei parametri dal server.")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        logger.info("Impostazione dei parametri ricevuti dal server.")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)
        logger.info("Parametri impostati con successo.")

    def fit(self, parameters, config):
        logger.info("Inizio del processo di fitting.")
        # Misurazione del tempo di comunicazione (ricezione dei pesi)
        comm_start = time.time()
        self.set_parameters(parameters)
        comm_end = time.time()
        communication_time = comm_end - comm_start
        logger.info(f"Tempo di comunicazione (ricezione): {communication_time:.4f} secondi.")

        # Misurazione del tempo di training
        train_start = time.time()
        train(self.net, self.trainloader, epochs=1)
        train_end = time.time()
        training_time = train_end - train_start
        logger.info(f"Tempo di training: {training_time:.4f} secondi.")

        # Misurazione del tempo di comunicazione (invio dei pesi)
        comm_start_2 = time.time()
        updated_parameters = self.get_parameters(config)
        comm_end_2 = time.time()
        communication_time += comm_end_2 - comm_start_2
        logger.info(f"Tempo di comunicazione (invio): {comm_end_2 - comm_start_2:.4f} secondi.")
        logger.info(f"Tempo totale di comunicazione: {communication_time:.4f} secondi.")

        # Restituzione dei tempi al server
        return updated_parameters, len(self.trainloader.dataset), {
            "training_time": training_time,
            "communication_time": communication_time
        }

    def evaluate(self, parameters, config):
        logger.info("Inizio del processo di valutazione.")
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        logger.info(f"Valutazione completata: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main():
    # Crea un'istanza del client Flower
    client = FlowerClient()
    # Avvia il client e connettiti al server
    logger.info("Avvio del client Flower e connessione al server.")
    fl.client.start_client(
        server_address="flwr_server:8080",  # Nome corretto del servizio Docker
        client=client.to_client()  # Utilizza .to_client() per convertire in Client
    )
    logger.info("Client Flower avviato correttamente.")

if __name__ == "__main__":
    max_retries = 5
    for i in range(max_retries):
        try:
            main()
            break  # Se la connessione ha successo, esci dal ciclo
        except Exception as e:
            logger.error(f"Tentativo {i+1}/{max_retries}: Errore nella connessione al server: {e}")
            time.sleep(5)  # Attendi 5 secondi prima di riprovare
    else:
        logger.critical("Impossibile connettersi al server dopo diversi tentativi.")
