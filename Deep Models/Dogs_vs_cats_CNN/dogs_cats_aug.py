import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import pickle  # Added this import

class DogsCatsAug():
    def __init__(self):
        self.data_from_kaggle = "D:\\Nural Networks\\ece5831-2023- HW's\\Kaggle_Data\\train"
        self.data_dirname = "D:\\Nural Networks\\ece5831-2023- HW's\\Dogs_vs_Cats"

        self.make_dataset("train", 0, 3001)
        self.make_dataset("validation", 7500, 8501)
        # total: 12,500
        self.make_dataset("test", 10000, 11001)

        self.batch_size = 32
        self.train_dataset = image_dataset_from_directory(f"{self.data_dirname}/train", image_size=(200, 180), batch_size=self.batch_size)
        self.validation_dataset = image_dataset_from_directory(f"{self.data_dirname}/validation", image_size=(200, 180), batch_size=self.batch_size)
        self.test_dataset = image_dataset_from_directory(f"{self.data_dirname}/test", image_size=(200, 180), batch_size=self.batch_size)

    def make_dataset(self, subset_name, start_idx, end_idx):
        for category in {"cat", "dog"}:
            self.dir = f"{self.data_dirname}\\{subset_name}\\{category}"
            os.makedirs(self.dir)
            fnames = [f"{category}.{i}.jpg" for i in range(start_idx, end_idx)]

            for fname in fnames:
                shutil.copyfile(src=f"{self.data_from_kaggle}\\{fname}", dst=f"{self.dir}\\{fname}")

    def aug_model():
        
        
        input = keras.Input(shape=(200, 180, 3))
        augmented_input = data_augmentation(input)


        x = layers.Rescaling(1./255)(augmented_input)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=input, outputs=outputs)
        return model

    def compile_model(self, model):
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def plot_images_with_predictions(self, model, generator, class_names):
        num_dog_images = 0
        num_cat_images = 0
        i = 0
        plt.figure(figsize=(12, 8))

        while num_dog_images < 10 or num_cat_images < 10:
            img_batch, true_labels_batch = generator[i % len(generator)]

            # Get predictions for the batch
            predictions_batch = model.predict(img_batch)

            for j in range(img_batch.shape[0]):
                img = np.expand_dims(img_batch[j], axis=0)
                true_label = true_labels_batch[j]

                # Get prediction for the current image
                predicted_label = class_names[np.argmax(predictions_batch[j])]

                # Check if the image is a dog or cat and if the quota is not reached
                if true_label == 1 and num_dog_images < 10:
                    num_dog_images += 1
                    # Plot dog image
                    plt.subplot(4, 5, num_dog_images)
                    plt.imshow(img.squeeze())
                    plt.title(f"True: {class_names[int(true_label)]}, Predicted: {predicted_label}")
                    plt.axis('off')
                elif true_label == 0 and num_cat_images < 10:
                    num_cat_images += 1
                    # Plot cat image
                    plt.subplot(4, 5, num_cat_images + 10)  # Start from the second row
                    plt.imshow(img.squeeze())
                    plt.title(f"True: {class_names[int(true_label)]}, Predicted: {predicted_label}")
                    plt.axis('off')

                if num_dog_images >= 10 and num_cat_images >= 10:
                    break

            i += 1

        plt.show()

    def save_model(self, model, filename):
        # Save the model using pickle
        with open(filename, 'wb') as file:
            pickle.dump(model, file)