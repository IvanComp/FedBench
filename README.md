# Open Science Artifact: Performance Analysis of Architectural Patterns for Federated Learning Systems

<p align="center">
<img src="img/logoFedBench.png" width="310px" height="230px"/>
</p>

<img src="https://img.shields.io/badge/version-1.0-green" alt="Version">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12671621.svg)](https://doi.org/10.5281/zenodo.12671621)

This Open Science Artifact contains a Federated Learning platform built on top of the [Flower](https://github.com/adap/flower) an open-source Python library that simplifies building Federated Learning systems.

This platform was utilized to in the paper "Performance Analysis of Architectural Patterns for Federated Learning Systems" for the IEEE International Conference on Software Architecture (ICSA 2025).

# Table of contents
<!--ts-->
   * [Abstract](#abstract)
   * [Package Structure](#packagestructure)
   * [Architectural Patterns](#architecturalpatterns)
   * [References](#references)
   
# Abstract

_Context_ Designing federated learning systems is not trivial, even more so when client devices show large heterogeneity while contributing to the learning process. Architectural patterns have been recently defined in the literature to deal with the design challenges of federated learning, thus providing reusable solutions to common problems within a given context. However, patterns lead to both benefits and drawbacks, e.g., introducing a client registry improves the maintainability but it requires extra-costs.    

_Objective_ The goal of this paper is to showcase the performance impact of applying architectural patterns in federated learning systems, thus pointing out the pros and cons of a selected number of three patterns.

_Method_ We extend the Flower framework, a well-assessed and unified approach to operationalize Federated Learning projects.

_Results_ Experimental results show evidence of the trade-off between system performance and learning accuracy, thus providing quantitative information to software architects and supporting them in selecting design alternatives. 

# Package Structure

The structure of this package is organized as follow:


# Architectural Patterns

The 4 Architectural Patterns proposed in [1] and implemented in our framework are:

| Architectural Pattern | Pattern Category | Description |
| --- | --- | --- | 
| **Client Registry** | `Client Management` | TODO |
| **Client Selector** | `Client Management` | TODO |
| **Client Cluster** | `Client Management` | TODO |
| **Message Compressor** | `Model Management` | TODO |

# References

[1] Sin Kit Lo, Qinghua Lu, Liming Zhu, Hye-Young Paik, Xiwei Xu, Chen Wang,
**Architectural patterns for the design of federated learning systems**,
Journal of Systems and Software, Volume 191, 2022, 111357.
