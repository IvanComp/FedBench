# Architectural Pattern for Federated Learning Systems

<p align="center">
<img src="img/ArchitectureEval.png" width="650px" height="350px"/>
</p>

This is the repository for testing the performance of a Federated Learning system applying custom specifications.
The Federated Learning system is supported by [Flower](https://github.com/adap/flower) Framework. 

The Flower framework was extended by adding 3 different Architectural Patterns for Clients Management [1]:

- 1 Client Registry.

- 2 Client Selector.

- 3 Client Clustering.

# Table of contents
<!--ts-->
   * [Functionalities](#functionalities)
   * [How to run](#how-to-run)
   * [References](#references)
   
# Functionalities

This repository is divided in two main folders:

- __1. FL using Multiple Models/__: Examples for running FL simulations considering multiple models

- __2. FL using Singles Models/__: Examples for running FL simulations considering singles models

# How To Run:

It's possible to run the framework in two ways. 

- Locally (creating virtual local images of clients and serves)

- Distributed (creating container images of clients and serves)

## To run the Simulation on Docker containers:

In the target with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 3 Client A, 3 Client B, Prometheus, Grafana)
NUM_ROUNDS=2 docker-compose up --scale clienta=2 --scale clientb=0
```

## To run the Simulation Locally:

In the target folder, enter the following command:

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 2 
```

Change the number of clients by modifying the value of the "--num-supernodes" variable.

# References

[1] Sin Kit Lo, Qinghua Lu, Liming Zhu, Hye-Young Paik, Xiwei Xu, Chen Wang,
**Architectural patterns for the design of federated learning systems**,
Journal of Systems and Software, Volume 191, 2022, 111357.