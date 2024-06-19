from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
