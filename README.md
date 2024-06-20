# **Federated Learning Architecture with Docker**
## I. Overview
This project builds an efficient federated learning architecture using Docker containers. Federated learning allows multiple clients to collaboratively train a machine learning model while keeping their data decentralized. The objective is to classify whether a user is a smoker or non-smoker based on 27 features, including sex and health state. The dataset format and information are illustrated in the provided figure.
---
## II. Architecture:
Horizontal Federated Learning: Since we have similar subsets of data, we are utilizing a horizontal federated learning approach.
FedAvg Method: For updating the global weights, we use the Federated Averaging (FedAvg) method, which calculates the new global model weights.

![fl](https://github.com/aymenhedi123/mini/assets/103534291/3f1cc670-ea06-4021-bafc-cd107b6a4b26)
---
## III. Docker Containers
We use three Docker containers:
Server Container: Manages the federated learning process and aggregates the model weights.
Client Containers (2): Each client trains the model locally with their respective subset of data and communicates with the server.
![a](https://github.com/aymenhedi123/mini/assets/103534291/9957db15-5858-4da2-9813-6b1714f3c423)
## VI. Requirements
We use the following libraries:

- TensorFlow: An open-source machine learning framework used for training and evaluating the models.
- Pandas: A data manipulation and analysis library.
- NumPy: A library for numerical computations.
- Flask: A micro web framework for Python used to ensure communication via API for sending weights.
---

## V. Script
### Docker Files
- Dockerfile.master:
- Dockerfile.client1:
- Dockerfile.client2:
  
### Python script:
#### master.py
- The server initializes the global training model weights (global_weight.h5),
- sends them to the clients, and waits for the clients to respond with updated weights. The server then applies the FedAvg method to update the global model weights and repeats the process until a stopping condition is met (maximum 20 iterations or loss < 0.2).

#### client1.py
- client1 receives the global weights
- It trains the model locally with its subset of data ; in our case it works with "smoking_subset_1.csv"
- It sends the updated weights and evaluation metrics back to the server

#### client2.py
- The client2 receives the global weights
- It trains the model locally with its subset of data ; in our case it works with "smoking_subset_2.csv"
- It sends the updated weights and evaluation metrics back to the server

### Docker Compose script:
The Docker Compose file defines and activates the three containers together, creating a network with a bridge driver and setting up the necessary volumes.

#### docker-compose.yml content

## VI. Processing
1.Clone the Repository:

git clone https://github.com/aymenhedi123/mini.git

2.Navigate to the Project Directory:

cd last/fl_two_client/mini 

3.Build and Run the Containers:

docker-compose up --build

** This README provides a clear and structured guide to your federated learning project, complete with explanations of the architecture, requirements, and step-by-step instructions for getting started.**




