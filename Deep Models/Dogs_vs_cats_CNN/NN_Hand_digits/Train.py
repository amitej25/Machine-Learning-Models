import mnist_keras

(x_train, y_train), (_, _) = mnist_keras.load_data()
model = mnist_keras.build_model()
model.fit(x_train, y_train, epochs=5)
model.save('model_amitej')