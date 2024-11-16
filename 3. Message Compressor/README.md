# Message Compressor Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/3MessageCompressor.ipynb) of the Message Compressor architectural pattern.

It is possible to replicate the same experiments proposed in the paper or run the experiments by considering different input parameters (see [section](#how-to-run-custom-experiments)).

```bash
$ tree .
.
├── /1. Without Message Compressor 
├── /2. With Message Compressor      
```

# Input Parameters

In the following, there are the input parameters for the Message Compressor architectural pattern.

| Parameter | Experiment Default Value | Description | 
| --- | --- | --- | 
| `NUM_ROUNDS` | 10 | Number of Federated Learning rounds. |
| `nS` | 1 | Number of Server. Container name: server|
| `nC` | 8 | Number of Clients. Container name: **client** |
| `n_CPU` | 1 | Number of physical CPU cores allocated to each container |
| `Model` | n | See other configurations [here](#model-parameters) |
| `RAM` | 2GB | Memory capacity allocated to each container |
| `Compression Algorithm` | zlib | The Compression/Decompression algorithm used is [zlib](https://github.com/madler/zlib) (1.3.1 version) |


# Model Parameters

| Parameter | Config. n/2 | Config. n | Config. n*2 |  
| --- | --- | --- | --- |
| `Conv. 1` | 3 filters, 5x5 kernel | 6 filters, 5x5 kernel | 12 filters, 5x5 kernel |
| `Pool` | Max pooling, 2x2 kernel | Max pooling, 2x2 kernel | Max pooling, 2x2 kernel |
| `Conv. 2` | 8 filters, 5x5 kernel | 16 filters, 5x5 kernel | 32 filters, 5x5 kernel |
| `FC 1` | 60 units | 120 units | 240 units |
| `FC 2` | 42 units | 84 units | 168 units |
| `FC 3` | 10 units | 20 units | 30 units |
| `Batch Size` | 32 | x | y |
| `Learning Rate` | 0.001 | 0.001 | 0.001 |
| `Optimizer` | SGD | SGD | SGD |

# How to run the Paper's Experiments

The commands for the experiments proposed in the paper are defined in the following.
To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Message Compressor_ or _Without Message Compressor_ folders based on the type of experiment.
3. Open the terminal and enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration 4a4b (1 Server, 8 Clients) 
NUM_ROUNDS=10 docker-compose up --scale client=8
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed because of the stochastic nature of the simulation**. 


# How to run Custom Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale client=20
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.
