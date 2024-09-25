# AP-for-FL

This is the repository for testing the performance of a Federated Learning environment.

## To run Local Simulation:

In the Federated Learning S Local folder, enter the following command:

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 2
```

Change the number of clients by modifying the number after supernodes

## To run the Simulation on Docker containers:

In the Federated Learning S with Docker folder, enter the following command:

```bash
docker-compose up --scale client=10
```

Change the number of clients by modifying the number of client


