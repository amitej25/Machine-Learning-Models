import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# We'll write individual functions rather than a class this time
# Defining a function to load Data From Kears mnist 
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

#Defining a Function To create a Model
def build_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Defining a Function for Training model
def train(model, model_name):
    (x_train, y_train), (_, _) = load_data()
    model.fit(x_train, y_train, epochs=5)
    model.save(f"{model_name}")
    return model

def load(model_name):
    return load_model(f"{model_name}")


# defining a function that tests the model
def test(model_name):
    (_, _), (x_test, y_test) = load_data()
    model = load(model_name)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

## predict function gives us the predcted value    

def predict(x, model_name):
    x = x.reshape(1, 784)
    model = load(model_name)
    prediction = model.predict(x)
    return np.argmax(prediction)
