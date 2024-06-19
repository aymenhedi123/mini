from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from threading import Lock
import numpy as np
import time

app = Flask(__name__)
lock = Lock()
clients_weights = []
clients_losses = []

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

@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.get_json()
    new_weights = [tf.convert_to_tensor(w) for w in data['weights']]
    model.set_weights(new_weights)
    print("Updated model weights received from client")
    return jsonify({'status': 'success', 'message': 'Model weights updated'})

@app.route('/update_weights', methods=['POST'])
def update_weights():
    data = request.get_json()
    local_weights = [tf.convert_to_tensor(w) for w in data['weights']]
    local_losses = data['losses']

    with lock:
        clients_weights.append(local_weights)
        clients_losses.append(local_losses)

    print("Client weights and losses received")
    return jsonify({'status': 'success', 'message': 'Client weights and losses received'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

def federated_averaging():
    global clients_weights, clients_losses

    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        time.sleep(10)  # Adjust sleep time based on your needs

        with lock:
            if clients_weights:
                new_weights = np.mean(clients_weights, axis=0)
                model.set_weights(new_weights)
                clients_weights = []  # Clear the weights for next iteration
                clients_losses = []   # Clear the losses for next iteration
                print("Global model updated with federated averaging")

        # Early stopping check
        if clients_losses and all(loss < 0.2 for loss in clients_losses):
            print("Early stopping criteria met. Stopping federated learning.")
            break

        iteration += 1

if __name__ == "__main__":
    import threading
    threading.Thread(target=federated_averaging, daemon=True).start()  # Start federated averaging thread

    app.run(host="0.0.0.0", port=5000)
