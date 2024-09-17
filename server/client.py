import socket
import pickle
import numpy as np

def train_local_model():
    # Simulazione di un modello addestrato localmente
    weights = np.random.rand(10)
    return weights

def send_weights_to_server(weights):
    host = 'server'
    port = 8080
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    data = pickle.dumps(weights)
    client_socket.send(data)
    client_socket.close()

def main():
    weights = train_local_model()
    print("Local weights:", weights)
    send_weights_to_server(weights)

if __name__ == "__main__":
    main()
