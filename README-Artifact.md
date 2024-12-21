# Open Science Artifact: Performance Analysis of Architectural Patterns for Federated Learning Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14039470.svg)](https://zenodo.org/uploads/14039470)

The replication packa can be found in the following zenodo repository: 
 
package contains scripts and results of the architectural patterns analyzed in the paper "Performance Analysis of Architectural Patterns for Federated Learning Systems" accepted for publication in the 22nd IEEE International Conference on Software Architecture (ICSA 2025). 

<tt>AP4FED</tt> is a Federated Learning Benchmark platform built on top of the [Flower](https://github.com/adap/flower) an open-source Python library that simplifies building Federated Learning systems. It enables the definition of customized Federated Learning system configurations by leveraging architectural patterns [1], aiming to extract and analyze system performance metrics.

<tt>AP4FED</tt> supports software architects by offering a framework for the quantitative evaluation of specific design decisions, enabling a deeper understanding of their impact on system performance and guiding the optimization of Federated Learning architectures.

<tt>AP4FED</tt> was utilized in the paper "_Performance Analysis of Architectural Patterns for Federated Learning Systems_" accepted for the 22nd IEEE International Conference on Software Architecture (ICSA 2025) [2].

# Table of contents
<!--ts-->
   * [Authors](#authors)
   * [Abstract](#abstract)
   * [Zenodo Package Structure](#zenodopackagestructure)
   * [Available Files](#availablefiles)
   * [Prerequisites](#prerequisites)
   * [How to Run](#how-to-run)
   
# Authors

Ivan Compagnucci (Gran Sasso Science Institute), Italy
Riccardo Pinciroli (Zimmer Biomet), Italy
Catia Trubiani (Gran Sasso Science Institute), Italy

# Abstract

**Context:** Designing Federated Learning systems is not trivial, as it requires managing heterogeneous and distributed clients' resources, while balancing data privacy and system efficiency. Architectural patterns have been recently specified in the literature to showcase reusable solutions to common problems within Federated Learning systems. However, patterns often lead to both benefits and drawbacks, e.g., introducing a message compressor algorithm may reduce the system communication time, but it may produce additional computational costs for clients' devices. 
**Objective:** The goal of this paper is to quantitatively investigate the performance impact of applying a selected set of architectural patterns when designing Federated Learning systems, thus providing evidence of their pros and cons. 
**Method:** We develop an open source environment by extending the well-established Flower framework; it integrates the implementation of four architectural patterns and evaluates their performance characteristics. **Results:** Experimental results assess that architectural patterns indeed bring performance gains and pains, as raised by the practitioners in the literature. Our framework can support software architects in making informed design choices when designing Federated Learning systems.

# Zenodo Package Structure

The structure of this package is organized as follow:

```bash
├── /1. Client Selector      # <-- contains `Client Selector` experiments
├── /2. Client Clustering    # <-- contains `Client Cluster` experiments
├── /3. Message Compressor   # <-- contains `Message Compressor` experiments
├── Experiments-Results.xlsx      # <-- contains experiments results proposed in the paper
└── README.md                # <-- README file
```

# Prerequisites

The experiments are designed to ensure accessibility and reproducibility, requiring only Docker to run experiments for extract performance metrics within a controlled and isolated environment.

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose (version v2.29.2 or higher)](https://docs.docker.com/compose/install/)

Make sure to have Docker and Docker Compose installed on your system to run this project.

- **Docker**: Required to create and run containers.
- **Docker Compose**: Enables running multi-container applications with Docker using the `docker-compose.yml` file.

You can verify the installation with the following commands:
```bash
docker --version
docker compose version
```

# Architectural Patterns

The 4 Architectural Patterns proposed in [1] and implemented in our framework are:

| Architectural Pattern | Pattern Category | Description | 
| --- | --- | --- | 
| **Client Selector** | `Client Management` | A mechanism to dynamically select clients based on specific criteria, such as data distribution, device capabilities, or network conditions, to optimize the federated learning process. |
| **Client Cluster** | `Client Management` | A strategy to group clients into clusters based on shared characteristics, such as data similarity or device attributes, improving model accuracy and convergence in non-IID data scenarios. | 
| **Message Compressor** | `Model Management` | A component designed to reduce the size of data exchanged between clients and the server by compressing messages, which lowers communication latency and bandwidth usage in federated learning environments. | 

The **Client Registry** architectural pattern is implemented by adding the following parameters:

| Attribute | Data Type | Description |
| --- | --- | --- | 
| **cID** | `string` | Client’s Unique Identifier | 
| **cluster_Type** | `string` | Cluster associated to the Client | 
| **n_CPU** | `int` | Number of Client’s CPU | 
| **training_time** | `float` | Client’s Training Time | 
| **communication_time** | `float` | Client’s Communication Time | 
| **total_round_time** | `float` | Client’s Total Round Time | 


# How To Run

Three distinct types of experiments can be conducted, each focusing on a different architectural pattern.

## 1. Client Selector Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/1ClientSelector.ipynb) of the Client Selector architectural pattern.

It is possible to replicate the same experiments proposed in the paper or run the experiments by considering different input parameters (see [section](#how-to-run-custom-experiments)).

```bash
$ tree .
.
├── /With Client Selector  
├── /Without Client Selector      
```

### Input Parameters

In the following, there are the input parameters for the Client Selector architectural pattern.

| Parameter | Experiment Default Value | Description | 
| --- | --- | --- | 
| `NUM_ROUNDS` | 10 | Number of Federated Learning rounds. |
| `nS` | 1 | Number of Server. Container name: server|
| `nC` | 4 | Number of Clients. Container name: **clienthigh** = High-Spec clients, <br> **clientlow** = Low-Spec clients|
| `n_CPU` | 2 for High-Spec clients,<br> 1 for Low-Spec clients | Number of physical CPU cores allocated to each container |
| `RAM` | 2GB | Memory capacity allocated to each container |
| `Selection Strategy` | Resource-based | The Selection Strategy will include/exclude clients based on their computational capabilities |
| `Selection Criteria` | Number of CPU > 1 | The Selector Criteria will evaluate clients based on their number of CPU |

### How to run the Paper's Experiments

The commands for the experiments proposed in the paper are defined in the following.
To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Client Selector_ or _Without Client Selector_ folders based on the type of experiment.
3. Open the terminal and enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration A (1 Server, 4 Clients with "High" specifications) -- Without Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=4 --scale clientlow=0

#For Configuration B (1 Server, 3 Clients A with "High" specifications, 1 Client with "Low" specifications) -- Without Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=3 --scale clientlow=1

#For Configuration C (1 Server, 3 Clients A with "High" specifications, 1 Client A with "Low" specifications) -- With Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=3 --scale clientlow=1
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed because of the stochastic nature of the simulation**. 


### How to run Custom Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale clienthigh=15 --scale clientlow=5
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.


## 2. Client Cluster

# Client Cluster Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/2ClientCluster.ipynb) of the Client Cluster architectural pattern.

It is possible to replicate the same experiments proposed in the paper or run the experiments by considering different input parameters (see [section](#how-to-run-custom-experiments)).

```bash
$ tree .
.
├── /1. With Client Cluster  
├── /2. Without Client Cluster      
```

### Input Parameters

In the following, there are the input parameters for the Client Cluster architectural pattern.

| Parameter | Experiment Default Value | Description | 
| --- | --- | --- | 
| `NUM_ROUNDS` | 10 | Number of Federated Learning rounds. |
| `nS` | 1 | Number of Server. Container name: server|
| `nC` | 8 | Number of Clients. Container name: **clienta** = IID clients, <br> **clientb** = non-IID clients|
| `n_CPU` | 1 | Number of physical CPU cores allocated to each container |
| `RAM` | 2GB | Memory capacity allocated to each container |
| `Clustering Strategy` | Data-based | The Clustering Strategy will groups clients based on their data distribution type |
| `Clustering Criteria` | IID vs non-IID | The Clustering Criteria will cluster clients based on IID or non-IID data |

### How to run the Paper's Experiments

The commands for the experiments proposed in the paper are defined in the following.
To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Client Cluster_ or _Without Client Cluster_ folders based on the type of experiment.
3. Open the terminal and enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration 4a4b (1 Server, 4 Clients with IID, 4 Clients with non-IID) 
NUM_ROUNDS=10 docker-compose up --scale clienta=4 --scale clientb=4

#For Configuration 5a3b (1 Server, 5 Clients with IID, 3 Clients with non-IID) 
NUM_ROUNDS=10 docker-compose up --scale clienta=5 --scale clientb=3

#For Configuration 6a2b (1 Server, 6 Clients with IID, 2 Clients with non-IID) 
NUM_ROUNDS=10 docker-compose up --scale clienta=6 --scale clientb=2
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed because of the stochastic nature of the simulation**. 


### How to run Custom Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale clienta=15 --scale clientb=5
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.

## 3. Message Compressor Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/3MessageCompressor.ipynb) of the Message Compressor architectural pattern.

It is possible to replicate the same experiments proposed in the paper or run the experiments by considering different input parameters (see [section](#how-to-run-custom-experiments)).

```bash
$ tree .
.
├── /1. With Message Compressor 
├── /2. Without Message Compressor      
```

### Input Parameters

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


### Model Parameters

| Parameter | Config. n/2 | Config. n | Config. n*2 |  
| --- | --- | --- | --- |
| `Conv. 1` | 3 filters, 5x5 kernel | 6 filters, 5x5 kernel | 12 filters, 5x5 kernel |
| `Pool` | Max pooling, 2x2 kernel | Max pooling, 2x2 kernel | Max pooling, 2x2 kernel |
| `Conv. 2` | 8 filters, 5x5 kernel | 16 filters, 5x5 kernel | 32 filters, 5x5 kernel |
| `FC 1` | 60 units | 120 units | 240 units |
| `FC 2` | 42 units | 84 units | 168 units |
| `FC 3` | 10 units | 20 units | 30 units |
| `Batch Size` | 32 | 32 | 32 |
| `Learning Rate` | 0.001 | 0.001 | 0.001 |
| `Optimizer` | SGD | SGD | SGD |

### How to run the Paper's Experiments

The commands for the experiments proposed in the paper are defined in the following.
To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Message Compressor_ or _Without Message Compressor_ folders based on the type of experiment.
3. Open the terminal and enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration 8 Clients (1 Server, 8 Clients) 
NUM_ROUNDS=10 docker-compose up --scale client=8
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed because of the stochastic nature of the simulation**. 


### How to run Custom Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale client=20
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.
