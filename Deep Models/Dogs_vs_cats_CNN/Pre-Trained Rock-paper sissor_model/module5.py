import sys
import glob
from mnist1 import Mnist

import glob
import mnist1 as mnist1
import numpy as np



# Use glob to find all matching image files
#image_files = glob.glob(image_pattern)
def main(image_pattern="*.png", digit=0):
    mnist = mnist1.Mnist()
    mnist.init_network()  # Initialize the neural network from the provided model file

    image_files = glob.glob(image_pattern)

    for image_filename in image_files:
        image = mnist.load_image_jpg(image_filename)  # Load the .jpg image
        predicted_digit = mnist.predict(image)
        predicted_digit = np.argmax(predicted_digit)
        if (predicted_digit == digit).any():
            print(f"Success: Image {image_filename} is for digit {digit} and is recognized as {predicted_digit}.")
        else:
            print(f"Fail: Image {image_filename} is for digit {digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        print("Usage: python module5.py <image_pattern> <digit>")
    else:
        main(sys.argv[1], int(sys.argv[2]))


