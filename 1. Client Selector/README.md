# Client Selector Architetcural Pattern

This folder contains models, scripts, data, and [link](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/1ClientSelector.ipynb) to the results of the Client Selector architectural pattern experiment.

# How to run Experiments

To reproduce the results presented in the paper, users can run new simulations. 
<br>
Note that, **different values may be observed due to the stochastic simulation**. Please, follow these steps:

1. Run Docker Desktop
2. Open the terminal in the _1. Client Selector_ folder
3. Enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration.
#In this example: 5 images (1 Server, 2 Clients A with "High" specifications, 2 Client A with "Low" specifications
NUM_ROUNDS=10 docker-compose up --scale clientahigh=2 --scale clientalow=2
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.

# Input Parameters

In the following, there are the input parameters for the Client Selector architectural pattern.

| Parameter | Experiment Default Value | Docker Container Name | Description | 
| --- | --- | --- | --- |
| `NUM_ROUNDS` | 10 | - | Number of Federated Learning rounds. |
| `nS` | 1 (Fixed) | - | Number of Server. |
| `nC` | 4 | **clienthigh** = High-Spec clients, <br> **clientlow** = Low-Spec clients | Number of Clients. |
| `n_CPU` | 2 for High-Spec clients,<br> 1 for Low-Spec clients | - | Number of physical CPU cores allocated to each container |
| `RAM` | 2GB | - | Memory capacity allocated to each container |


# Customize Input Parameters

Users can also tune the input parameters of provided to investigate performance using with the same pattern.
All of the following input parameters can be varied by changing the corresponding values from the command line before starting the project.


Due to the physical allocation of CPU and RAM, changing these parameters requires access to the **docker-compose** file in which the resources to be allocated to the container are set.


