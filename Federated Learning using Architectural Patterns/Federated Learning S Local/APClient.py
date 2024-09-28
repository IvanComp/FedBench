from datetime import datetime
import hashlib

# Client Registry with hashing
class ClientRegistry:
    def __init__(self):
        self.registry = {}

    def hash_client_id(self, client_info):
        """Generates a unique hash for the client based on its information."""
        client_string = f"{client_info['ip']}-{client_info['port']}"
        return hashlib.sha256(client_string.encode()).hexdigest()

    def register_client(self, client_id, resource_info):
        """Registers a new client with its ID and resources"""
        self.registry[client_id] = {
            'resources': resource_info,
            'active': True,
            'last_update': datetime.now()
        }
        print(f"Client {client_id} registered with resources: {resource_info}")

    def update_client(self, client_id, status):
        """Updates the status of the client."""
        if client_id in self.registry:
            self.registry[client_id]['active'] = status
            self.registry[client_id]['last_update'] = datetime.now()
            print(f"Client {client_id} status updated: {status}")

    def remove_client(self, client_id):
        """Removes a client from the registry."""
        if client_id in self.registry:
            del self.registry[client_id]
            print(f"Client {client_id} removed from registry")

    def get_active_clients(self):
        """Returns the list of active clients."""
        active_clients = {client_id: info for client_id, info in self.registry.items() if info['active']}
        print(f"Active clients: {list(active_clients.keys())}")
        return active_clients
