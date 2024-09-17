import socket
import pickle
import numpy as np

def aggregate_weights(client_weights):
    # Media dei pesi ricevuti dai client
    return np.mean(client_weights, axis=0)

def main():
    host = '0.0.0.0'
    port = 8080
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    print("Server listening on port 8080...")

    client_weights = []
    while True:
        client_conn, addr = server_socket.accept()
        data = client_conn.recv(4096)
        if data:
            weights = pickle.loads(data)
            client_weights.append(weights)
            if len(client_weights) == 3:  # Supponiamo di avere 3 client
                updated_weights = aggregate_weights(client_weights)
                print("Aggregated weights:", updated_weights)
                client_weights = []

        client_conn.close()

if __name__ == "__main__":
    main()
