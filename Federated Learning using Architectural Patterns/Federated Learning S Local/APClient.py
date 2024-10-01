import platform
import psutil
from datetime import datetime

class ClientRegistry:
    def __init__(self):
        self.registry = {}

    def get_client_info(self):

        cpu_count = psutil.cpu_count(logical=False)  # Physical cores

        # Logica di assegnazione dinamica del cluster
        if cpu_count <= 1:
            assigned_cluster = 'A'
        else:
            assigned_cluster = 'B'

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
            "python_version": platform.python_version(),
            "cluster": assigned_cluster,
        }
        return info

    def register_client(self, cid, resource_info):
        """Registers a new client with its cid and detailed system information."""
        client_info = self.get_client_info()
        self.registry[cid] = {
            'resources': resource_info,
            'system_info': client_info,  # Save the system info
            'active': True,
            'last_update': datetime.now()
        }
        print(f"Client {cid} registered with resources: {resource_info}")
        print(f"System info: {client_info}")

    def update_client(self, cid, status, last_update=None):
        """Updates the status of the client."""
        if cid in self.registry:
            self.registry[cid]['active'] = status
            self.registry[cid]['last_update'] = last_update or datetime.now()
            print(f"Client {cid} status updated: {status}")

    def get_active_clients(self):
        """Returns the list of active clients."""
        active_clients = {cid: info for cid, info in self.registry.items() if info['active']}
        print(f"Active clients: {list(active_clients.keys())}")
        return active_clients

    def is_registered(self, cid):
        """Check if a client is already registered."""
        return cid in self.registry

    def print_clients_info(self):
        """Prints the list of clients with their respective system information."""
        active_clients = self.get_active_clients()
        if active_clients:
            for i, (cid, client_info) in enumerate(active_clients.items(), 1):
                system_info = client_info['system_info']
                print(f"Client {i}: {cid}")
                for key, value in system_info.items():
                    print(f"  {key}: {value}")
                print()  # Newline for readability
        else:
            print("No active clients registered.")
