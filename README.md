## Federated Learning Architecture with Docker
# Overview:

This project builds an efficient federated learning architecture using Docker containers. Federated learning allows multiple clients to collaboratively train a machine learning model while keeping their data decentralized. The objective is to classify whether a user is a smoker or non-smoker based on 27 features, including sex and health state. The dataset format and information are illustrated in the provided figure.
# Architecture:
Horizontal Federated Learning: Since we have similar subsets of data, we are utilizing a horizontal federated learning approach.
FedAvg Method: For updating the global weights, we use the Federated Averaging (FedAvg) method, which calculates the new global model weights.

![fl](https://github.com/aymenhedi123/mini/assets/103534291/3f1cc670-ea06-4021-bafc-cd107b6a4b26)

# Docker Containers
We use three Docker containers:
Server Container: Manages the federated learning process and aggregates the model weights.
Client Containers (2): Each client trains the model locally with their respective subset of data and communicates with the server.
Docker Images
Server Docker File:
dockerfile
# Dockerfile content for the server
img

Client Docker File (identical for both clients except for the data):
dockerfile
# Dockerfile content for the clients
img
# Requirements
We use the following libraries:
TensorFlow: An open-source machine learning framework used for training and evaluating the models.
Pandas: A data manipulation and analysis library.
NumPy: A library for numerical computations.
Flask: A micro web framework for Python used to ensure communication via API for sending weights, etc.
How It Works
# master.py:
 The master.py script initializes the global training model weights (global_weight.h5), sends them to the clients, and waits for the clients to respond with updated weights. The server then applies the FedAvg method to update the global model weights and repeats the process until a stopping condition is met (maximum 20 iterations or loss < 0.2).
python
# master.py code:
img
# client.py
The client.py script receives the global weights, trains the model locally with its subset of data, and sends the updated weights and evaluation metrics back to the server.
python
# client.py code
img
# Docker Compose
The Docker Compose file defines and activates the three containers together, creating a network with a bridge driver and setting up the necessary volumes.
yaml
# docker-compose.yml content
img
# Steps to Begin the Process
1.Clone the Repository:

git clone https://github.com/aymenhedi123/mini.git

2.Navigate to the Project Directory:

cd last/fl_two_client/mini 

3.Build and Run the Containers:

docker-compose up --build

This README provides a clear and structured guide to your federated learning project, complete with explanations of the architecture, requirements, and step-by-step instructions for getting started.




