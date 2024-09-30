import platform
import psutil
import hashlib
from datetime import datetime

class ClientRegistry:
    def __init__(self):
        self.registry = {}

    def get_client_info(self):
        """Gathers system information about the client."""
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),  # Physical cores
            "cpu_threads": psutil.cpu_count(logical=True),  # Logical threads
            "ram_total": psutil.virtual_memory().total / (1024 ** 3),  # Total RAM in GB
            "ram_available": psutil.virtual_memory().available / (1024 ** 3),  # Available RAM in GB
        }
        return info

    def hash_client_id(self, client_info):
        """Generates a unique hash for the client based on its information."""
        client_string = f"{client_info['platform']}-{client_info['cpu']}-{client_info['ram_total']}"
        return hashlib.sha256(client_string.encode()).hexdigest()

    def register_client(self, client_id, resource_info):
        """Registers a new client with its ID and detailed system information."""
        client_info = self.get_client_info()  # Gather additional system info
        self.registry[client_id] = {
            'resources': resource_info,
            'system_info': client_info,  # Save the system info
            'active': True,
            'last_update': datetime.now()
        }
        print(f"Client {client_id} registered with resources: {resource_info}")
        print(f"System info: {client_info}")

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
