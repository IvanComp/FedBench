# FedBench

<p align="center">
<img src="img/logoFedBench.png" width="280px" height="130px"/>
</p>

<img src="https://img.shields.io/badge/version-1.0-green" alt="Version">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12671621.svg)](https://doi.org/10.5281/zenodo.12671621)

FedBench is a Federated Learning platform built on top of the [Flower](https://github.com/adap/flower) an open-source Python library that simplifies building Federated Learning systems.
FedBench is a Federated Learning framework enhanced with architectural patterns and extended monitoring capabilities for performance evaluation.

## Authors

Ivan Compagnucci - Gran Sasso Science Institute (Italy)<br/>
Riccardo Pinciroli - Gran Sasso Science Institute (Italy)<br/>
Catia Trubiani - Gran Sasso Science Institute (Italy)

# Table of contents
<!--ts-->
   * [Functionalities](#functionalities)
   * [How to run](#how-to-run)
   * [Experiments](#experiments)
   * [Architectural Patterns](#architecturalpatterns)
   * [Performance](#performance)
   * [References](#references)
   
# Experiments

This section proposes the replication package of the experiments for the paper titled "Performance Analysis of Architectural Patterns
for Federated Learning Systems" for the IEEE International Conference on Software Architecture (ICSA 2024).

## To run the Framework on Docker containers:

# Client Selector

In the __1. Client Selector/__ folder with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 2 Clients A with "High" specifications, 2 Client A with "Low" specifications
NUM_ROUNDS=10 docker-compose up --scale clientahigh=2 --scale clientalow=2
```

# Client Cluster

In the __2. Client Cluster/__ folder with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 2 Clients A, 2 Clients B)
NUM_ROUNDS=10 docker-compose up --scale clienta=2 --scale clientb=2
```

# Message Compressor

In the __3. Message Compressor/__ folder with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 5 Client)
NUM_ROUNDS=10 docker-compose up --scale client=5
```

# Functionalities

FedBench allows you to 

# How To Run:

FedBench allows to run a Federated Learning project in two different ways:

- Locally (creating virtual local images of clients and server)

- Distributed (creating container images of clients and server)

# Architectural Patterns

FedBench extends the Flower framework by adding 4 Architectural Patterns proposed in [1]:

| Architectural Pattern | Pattern Category | Description |
| --- | --- | --- | 
| **Client Registry** | `Client Management` | TODO |
| **Client Selector** | `Client Management` | TODO |
| **Client Cluster** | `Client Management` | TODO |
| **Message Compressor** | `Model Management` | TODO |

# Performance

FedBench allows to generation of a set of performance benchmarks (graphs) derived from the execution.
They are automatically stored in the _/performance_ folder. Below you can find some examples.



# References

[1] Sin Kit Lo, Qinghua Lu, Liming Zhu, Hye-Young Paik, Xiwei Xu, Chen Wang,
**Architectural patterns for the design of federated learning systems**,
Journal of Systems and Software, Volume 191, 2022, 111357.
