# Client Selector Architetcural Pattern

This folder contains models, scripts, data, and link to the results of the Client Selector architectural pattern experiment.

# How to run Experiments

To reproduce results presented in the paper, users can run new simulation. 
In this case, they should note that slightly different values may be observed due to the stochastic simulation. Please, follow these steps:

1.
2.

# Customize Input Parameters

Users can also tune input parameters of provided models to study different applications designed with the same patter.
Please, follow these steps to study your own applications with our framework:


In the __1. Client Selector/__ folder with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 2 Clients A with "High" specifications, 2 Client A with "Low" specifications
NUM_ROUNDS=10 docker-compose up --scale clientahigh=2 --scale clientalow=2
```
