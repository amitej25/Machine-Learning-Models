import gzip
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image

class Mnist():
    img_size = 28*28
    model_file_name = "D:/Nural Networks/ece5831-2023- HW's/Assignment 5/sample_weight.pkl"
    key_file = {
        'test_img':     'Mnist\\t10k-images-idx3-ubyte (1).gz',
        'test_label':   'Mnist\\t10k-labels-idx1-ubyte (1).gz'
    }

    def __init__(self):
        self.network = None

    def load_image_jpg(self,file_name):
        image = Image.open(file_name).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = image_array.reshape(-1,self.img_size)
        return image_array


    def sigmoid(self,a):
        return 1/(1 + np.exp(-a))


    def softmax(self, a):
        c = np.max(a)
        a = np.exp(a-c)
        s = np.sum(a)

        return a


    def init_network(self):
        with open(self.model_file_name, 'rb') as f:
            self.network = pickle.load(f)

        return self.network  
       


    def predict(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3

        y =  self.softmax(a3)
    
        return y 