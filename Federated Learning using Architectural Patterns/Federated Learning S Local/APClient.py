import hashlib
from datetime import datetime

# Client Registry with hashing
class ClientRegistry:
    def __init__(self):
        self.registry = {}

    def is_client_registered(self, client_id):
        """Check if the client is already registered."""
        return client_id in self.registry

    def get_client_resources(self, client_id):
        """Retrieve the resources of a registered client."""
        if client_id in self.registry:
            return self.registry[client_id]['resources']
        return None

    def register_client(self, client_id, resource_info):
        """Register a new client with its resources."""
        self.registry[client_id] = {
            'resources': resource_info,
            'active': True,
            'last_update': datetime.now()
        }
        print(f"Client {client_id} registered with resources: {resource_info}")

    def update_client(self, client_id, resource_info):
        """Update the resources of a client."""
        if client_id in self.registry:
            self.registry[client_id]['resources'] = resource_info
            self.registry[client_id]['last_update'] = datetime.now()
            print(f"Client {client_id} updated with new resources: {resource_info}")

    def remove_client(self, client_id):
        """Remove a client from the registry."""
        if client_id in self.registry:
            del self.registry[client_id]
            print(f"Client {client_id} removed from registry")
