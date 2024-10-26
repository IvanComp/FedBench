# FedBench

<img src="https://img.shields.io/badge/version-1.0-green" alt="Version">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12671621.svg)](https://doi.org/10.5281/zenodo.12671621)


FedBench is a Federated Learning platform built on top of the [Flower](https://github.com/adap/flower) an open-source Python library that simplifies building Federated Learning systems.
FedBench is a Federated Learning framework enhanced with architectural patterns and extended monitoring capabilities for performance evaluation.

# Table of contents
<!--ts-->
   * [Functionalities](#functionalities)
   * [How to run](#how-to-run)
   * [Experiments](#experiments)
   * [Architectural Patterns](#architecturalpatterns)
   * [Performance](#performance)
   * [References](#references)
   
# Functionalities

This repository is divided in two main folders:

- __1. Client Selector/__: Running FL simulations considering the Client Selector pattern

- __2. Client Clustering/__: Running FL simulations considering the Client Clustering pattern

- __3. Message Compressor/__: Running FL simulations considering the Message Compressor pattern

# How To Run:

It's possible to run the framework in two ways. 

- Locally (creating virtual local images of clients and serves)

- Distributed (creating container images of clients and serves)


# Functionalities

This section propose the replication package of the experiments for the paper titled "Performance Analysis of Architectural Patterns
for Federated Learning Systems" for the IEEE International Conference on Software Architecture (ICSA 2024).

## To run the Simulation on Docker containers:

# Client Selector

In the __1. Client Selector/__ folder with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 2 Clients A with "High" specifications, 2 Client A with "Low" specifications
NUM_ROUNDS=10 docker-compose up --scale clientahigh=2 --scale clientalow=2
```

# Client Clustering

In the __2. Client Clustering/__ folder with 'docker-compose' file, enter the following command:

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

## To run the Simulation Locally:

In the target folder, enter the following command:

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 2 
```
Change the number of clients by modifying the value of the "--num-supernodes" variable.

# Architectural Patterns

The Flower framework was extended by adding 4 Architectural Patterns proposed in [1]:

- 1 Client Registry.

- 2 Client Selector.

- 3 Client Clustering.

- 4 Message Compressor.

# Performance

FedBench allows to generate a set of performance benchmarks (graphs) derived from the execution.
They are automatically stored in the _/performance_ folder.

# References

[1] Sin Kit Lo, Qinghua Lu, Liming Zhu, Hye-Young Paik, Xiwei Xu, Chen Wang,
**Architectural patterns for the design of federated learning systems**,
Journal of Systems and Software, Volume 191, 2022, 111357.
