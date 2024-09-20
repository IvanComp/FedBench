import flwr as fl

def main():
    # Definisci la strategia federata (opzionale)
    strategy = fl.server.strategy.FedAvg()

    # Avvia il server Flower
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
