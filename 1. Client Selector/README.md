# Client Selector Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/1ClientSelector.ipynb) of the Client Selector architectural pattern.

It is possible to replicate the same experiments proposed in the paper or change the settings of the experiment considering different input parameters.

The subfolder cpontaints the experiments discussed in the paper. For customized experiments see this [section](#how-to-run-customized-experiments)

```bash
$ tree .
.
├── /1. Without Client Selector   # <-- contains Configurations A and B to replicate experiments 
├── /2. With Client Selector      # <-- contains Configurations C to replicate experiment 
```

# Input Parameters

In the following, there are the input parameters for the Client Selector architectural pattern.

| Parameter | Experiment Default Value | Description | 
| --- | --- | --- | 
| `NUM_ROUNDS` | 10 | Number of Federated Learning rounds. |
| `nS` | 1 | Number of Server. Container name: server|
| `nC` | 4 | Number of Clients. Container name: **clienthigh** = High-Spec clients, <br> **clientlow** = Low-Spec clients|
| `n_CPU` | 2 for High-Spec clients,<br> 1 for Low-Spec clients | - | Number of physical CPU cores allocated to each container |
| `RAM` | 2GB | Memory capacity allocated to each container |
| `Selection Strategy` | Resource-based | The Selection Strategy will include/exclude clients based on their computational capabilities |
| `Selection Criteria` | Number of CPU > 1 | The Selector Criteria will evaluate clients based on their number of CPU |

# How to run the Paper's Experiments

To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Client Selector_ or _Without Client Selector_ folders based on the type of experiment.
2. Open the terminal in the _A_,_B_ or _C_ folders (Corresponding to the default configurations of the experiments proposed in the paper).
3. Based on the folder name, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration A (1 Server, 4 Clients with "High" specifications) -- Without Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clientahigh=4 --scale clientalow=0

#For Configuration B (1 Server, 3 Clients A with "High" specifications, 1 Client with "Low" specifications) -- Without Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clientahigh=3 --scale clientalow=1

#For Configuration C (1 Server, 3 Clients A with "High" specifications, 1 Client A with "Low" specifications) -- With Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=3 --scale clientlow=1
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed due to the stochastic simulation**. 


# How to run Customized Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale clienthigh=15 --scale clientlow=5
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.
