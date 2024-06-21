from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from threading import Lock
import numpy as np

app = Flask(__name__)
lock = Lock()
clients_weights = []
clients_losses = []
clients_updates = 0  # Track the number of clients that have sent updates
total_clients = 2  # Total number of clients expected to send updates

# Define and compile the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(24,)),  # Adjust input_shape as per your features
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the pre-trained weights
model.load_weights('global_weights.h5')
print("Loaded initial model weights from global_weights.h5")

@app.route('/get_model', methods=['GET'])
def get_model():
    weights = model.get_weights()
    weights_serializable = [w.tolist() for w in weights]  # Convert weights to list for serialization
    return jsonify({'weights': weights_serializable})

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global clients_updates
    print("Updated model weights received from clients")
    data = request.get_json()
    local_weights = [np.array(w) for w in data['weights']]
    local_loss = data['loss']
    local_accuracy = data['accuracy']

    with lock:
        clients_weights.append(local_weights)
        clients_losses.append(local_loss)
        clients_updates += 1

    print("Client weights and loss received")

    # Check if all clients have sent their updates for this iteration
    if clients_updates == total_clients:
        federated_averaging()
        clients_updates = 0  # Reset the counter for the next iteration

    return jsonify({'status': 'success', 'message': 'Client weights and loss received'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

def federated_averaging():
    global clients_weights, clients_losses

    with lock:
        if clients_weights:
            # Average each layer's weights across all clients
            new_weights = [np.mean([client_weights[layer] for client_weights in clients_weights], axis=0) for layer in range(len(clients_weights[0]))]
            model.set_weights(new_weights)
            print("Global model weights:\n", new_weights[0])
            clients_weights = []  # Clear the weights for next iteration
            clients_losses = []   # Clear the losses for next iteration
            print("Global model updated with federated averaging")

        # Early stopping check
        if all(loss < 0.1 for loss in clients_losses):
            print("Early stopping criteria met. Stopping federated learning.")
            model.save("final_model.h5")

if __name__ == "__main__":
    import threading
    threading.Thread(target=federated_averaging, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
