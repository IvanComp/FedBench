# Architectural Pattern for Federated Learning Systems

<p align="center">
<img src="img/ArchitectureEval.png" width="300px" height="150px"/>
</p>

This is the repository for testing the performance of a Federated Learning system applying custom specifications.

# Table of contents
<!--ts-->
   * [Functionalities](#functionalities)
   * [How to run](#how-to-run)
   * [References](#references)
   
# Functionalities

This repository is divided in two main folders:

- __Federated Learning without using Architectural Patterns/__:

- __Federated Learning using Architectural Patterns/__: Using clients architectural patterns.


# How To Run:

In both folders, you can use the following commands:

## To run the Simulation Locally:

In the Federated Learning S Local folder, enter the following command:

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 2 --backend-config {}
```

Change the **number of clients** by modifying the number after supernodes or add variables to **backend-config** to customize clients using a JSON schema e.g ‘{“<keyA>”:<value>, “<keyB>”:<value>}’

## To run the Simulation on Docker containers:

In the Federated Learning S with Docker folder, enter the following command:

```bash
docker-compose up --scale client=10
```

Change the number of clients by modifying the number of client


# References

[1] Sin Kit Lo, Qinghua Lu, Liming Zhu, Hye-Young Paik, Xiwei Xu, Chen Wang,
**Architectural patterns for the design of federated learning systems**,
Journal of Systems and Software, Volume 191, 2022, 111357.