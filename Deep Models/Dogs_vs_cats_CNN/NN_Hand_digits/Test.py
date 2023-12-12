import mnist_keras

(_, _), (x_test, y_test) = mnist_keras.load_data()
model = mnist_keras.load(model_name='model_amitej')
mnist_keras.test(model_name='model_amitej')
