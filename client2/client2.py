import requests
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import sleep

# Define the client model
model = Sequential([
    Dense(64, activation='relu', input_shape=(27,)),  # Adjust input_shape as per your features
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to fetch model weights with retries
def get_model_with_retries(url, max_retries=5, backoff=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                sleep(backoff)
            else:
                raise

# Function to send updated model weights to the master
def send_updated_model(url, updated_weights):
    weights_serializable = [w.tolist() for w in updated_weights]
    response = requests.post(url, json={'weights': weights_serializable})
    response.raise_for_status()
    print(f"Updated model sent to {url}, response: {response.json()}")

# Fetch model weights from master
if __name__ == "__main__":
    print("Reading data")
    df = pd.read_csv('smoking_subset_2.csv')  # Use 'smoking_subset_1.csv' for client1
    print(df.head())

    url = 'http://master:5000/get_model'
    response = get_model_with_retries(url)
    weights = response.json()['weights']
    weights = [tf.convert_to_tensor(w) for w in weights]  # Convert lists back to tensors
    model.set_weights(weights)
    print("Model weights set from the master server")

    # Prepare the data
    X = df.drop('target', axis=1).values  # Assuming 'target' is the label column
    y = df['target'].values  # Adjust to match your data structure

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model locally
    model.fit(X_train, y_train, epochs=1)  # Adjust epochs and batch size as needed
    print("Local training completed")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Send the updated model back to the master
    updated_weights = model.get_weights()
    send_updated_model('http://master:5000/update_model', updated_weights)
