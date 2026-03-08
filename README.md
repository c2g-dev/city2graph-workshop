# GeoAI in Practice: From Geospatial Data to Graph Neural Networks with City2Graph

![Workshop Thumbnail](img/workshop_thumbnail.jpg)

This workshop introduces Graph Neural Networks (GNNs) for geospatial practitioners. Using open-source Python tools including [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [City2Graph](https://city2graph.net/), participants will learn how to transform urban geospatial data into network structures and apply GNNs to model complex spatial relations.

## News
* **2026-03-08:** Repository updated for the upcoming workshop in **FOSS4G 2026 Hiroshima**.

## Description
As Geospatial Artificial Intelligence (GeoAI) evolves, Graph Neural Networks (GNNs) have emerged as a promising approach for predicting and understanding complex spatial relationships. This workshop provides a practical overview of the full GNN pipeline, from processing raw spatial data to training models, using an open-source Python stack: GeoPandas, NetworkX, PyTorch Geometric, OSMnx, and City2Graph.

## Who is this for?
* **Target Audience:** GIS analysts, spatial data scientists, and Python developers expanding their GeoAI and network modelling skills.
* **Prerequisites:** Basic proficiency in Python (especially GeoPandas) and GIS concepts. Basic knowledge of neural networks is highly recommended (e.g., activation functions, backpropagation, loss functions, etc.). No prior network science (NetworkX) or GNN (PyTorch Geometric) skills required.


## Contents

**Part 1: Graph Data Engineering, Spatial Network Analysis, and GNNs**

Learn to construct and analyse spatial networks using GeoPandas and NetworkX. We will demonstrate how to convert standard geospatial data (e.g., OpenStreetMap, GTFS, etc.) into unified graph structures with OSMnx and City2Graph. We will then explore key GNN architectures and transition from spatial graphs into tensor formats using PyTorch Geometric and City2Graph.

**Part 2: Build Your Own GeoAI Pipeline**

Put your skills into practice. Choose a city, extract its street network from OpenStreetMap or Overture Maps (optional), and train a Graph Autoencoder (GAE) for an unsupervised spatial clustering task. We will conclude by discussing how the GNN pipeline could be adopted for your business / research workflows.