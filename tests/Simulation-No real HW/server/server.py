import flwr as fl
import csv
import matplotlib.pyplot as plt

# Variabili per registrare i tempi
training_times = []
communication_times = []
rounds = []

def aggregate_fit(rnd, results, failures):
    global training_times, communication_times, rounds
    total_training_time = 0
    total_communication_time = 0

    # Aggrega i tempi di training e comunicazione di tutti i client
    for res in results:
        total_training_time += res.metrics["training_time"]
        total_communication_time += res.metrics["communication_time"]

    # Salva i tempi per ogni round
    rounds.append(rnd)
    training_times.append(total_training_time)
    communication_times.append(total_communication_time)

    # Continua con l'aggregazione normale
    return fl.server.strategy.FedAvg.aggregate_fit(rnd, results, failures)

def save_report():
    # Salva i tempi in un file CSV
    with open("report.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Training Time (s)", "Communication Time (s)"])
        for i in range(len(rounds)):
            writer.writerow([rounds[i], training_times[i], communication_times[i]])

    # Crea il grafico dei tempi
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, training_times, label='Training Time', marker='o')
    plt.plot(rounds, communication_times, label='Communication Time', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Time (s)')
    plt.title('Training and Communication Time per Round')
    plt.legend()
    plt.grid(True)
    plt.savefig('report_plot.png')  # Salva il grafico come immagine

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=aggregate_fit,
    )

    # Avvia il server Flower
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Dopo che tutte le round sono complete, salva il report
    save_report()

if __name__ == "__main__":
    main()
