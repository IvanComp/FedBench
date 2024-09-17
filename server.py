from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

# Definizione della strategia di Federated Learning
strategy = FedAvg(
    fraction_fit=1.0,  # Usa il 100% dei client per il training
    fraction_evaluate=0.5,  # Valuta il 50% dei client
    min_fit_clients=10,  # Numero minimo di client per il training
    min_evaluate_clients=5,  # Numero minimo di client per la valutazione
    min_available_clients=10,  # Numero minimo di client attivi per iniziare
)

# Funzione per la creazione del ServerApp
def server_fn(context):
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)

# Creazione e avvio del server
if __name__ == "__main__":
    server = ServerApp(server_fn=server_fn)
    server.run()
