#librarys
import keras
import wandb
from wandb.keras import WandbCallback
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from keras.utils import to_categorical

#Configuration
run = wandb.init()
config = run.config
###
config.dense_layer_size = 2
config.epochs = 50
config.optimizer = "adam"
config.hidden_nodes = 100




##########################################################################
################# Putting my data in the correct format ##################
####################### like Keras for tensorflow ########################
##########################################################################

import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def load_and_process_images(folder_path, label):
    images = []
    labels = []

    # Use glob to get all PNG files in the specified folder
    image_paths = glob.glob(f"{folder_path}/*.png")

    for path in image_paths:
        img = Image.open(path)
        img_array = np.array(img)

        images.append(img_array)
        labels.append(label)

    return np.array(images), np.array(labels)

# Specify the paths to the train, validation, and test folders
train_folder = 'D:/SMALL_IMAGES_CNN/train'
val_folder = 'D:/SMALL_IMAGES_CNN/val'
test_folder = 'D:/SMALL_IMAGES_CNN/test'

# Load and process training images
x_train_class1, y_train_class1 = load_and_process_images(f'{train_folder}/positive', label=1)
x_train_class2, y_train_class2 = load_and_process_images(f'{train_folder}/negative', label=0)

# Load and process testing images
x_test_class1, y_test_class1 = load_and_process_images(f'{test_folder}/positive', label=1)
x_test_class2, y_test_class2 = load_and_process_images(f'{test_folder}/negative', label=0)

# Load and process val images
x_val_class1, y_val_class1 = load_and_process_images(f'{val_folder}/positive', label=1)
x_val_class2, y_val_class2 = load_and_process_images(f'{val_folder}/negative', label=0)


# Concatenate the arrays for each class
x_train = np.concatenate([x_train_class1, x_train_class2], axis=0)
y_train_raw = np.concatenate([y_train_class1, y_train_class2], axis=0)
x_test = np.concatenate([x_test_class1, x_test_class2], axis=0)
y_test_raw = np.concatenate([y_test_class1, y_test_class2], axis=0)
x_val = np.concatenate([x_val_class1, x_val_class2], axis=0)
y_val_raw = np.concatenate([y_val_class1, y_val_class2], axis=0)

# Assuming you have class names as a list
class_names = ['negative','positive']


# Removing the last dimension, that is only 255 values
x_test = x_test[:, :, :, 0]
x_train = x_train[:, :, :, 0]
x_val = x_val[:, :, :, 0]

# Image pixels
img_width = x_train.shape[1]
img_height = x_train.shape[2]

# One hot encode output
y_train = to_categorical(y_train_raw)
y_val = to_categorical(y_val_raw)
y_test = to_categorical(y_test_raw)
labels = range(1)
num_classes = y_train.shape[1]

# Normalizing the pixel values
x_train_normalized = x_train / 255.
x_val_normalized = x_val / 255.
x_test_normalized = x_test / 255.

# Load ResNet50 Trained on imagenet
resnet_model = ResNet50(weights="imagenet")

# Preprocess the images the same way resnet images were preprocessed
x_train_preprocessed = preprocess_input(x_train)
x_val_preprocessed = preprocess_input(x_val)
x_test_preprocessed = preprocess_input(x_test)

# Build a new model that is ResNet50 minus the very last layer
last_layer = resnet_model.get_layer("avg_pool")
resnet_layers = keras.Model(inputs=resnet_model.inputs, outputs=last_layer.output)

###################################################################################################
#Here it is predicting because this is not the final model that we are going to use in our data set
###################################################################################################
# Use the resnet to "predict" but because we have removed the top layer,
# this outputs the activations of the second to last layer on our dataset
x_train_features = resnet_layers.predict(x_train_preprocessed)
x_val_features = resnet_layers.predict(x_val_preprocessed)
x_test_features = resnet_layers.predict(x_test_preprocessed)

# Creating the new model with the ResNet Layers
feature_model = Sequential()
feature_model.add(Dense(config.dense_layer_size, activation="sigmoid"))
feature_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


###################################################################################################
#Here the official training begins, with the learned features from .predict before
###################################################################################################
# Training the new model with validation set
feature_model.fit(x_train_features, y_train, epochs=config.epochs, validation_data=(x_val_features, y_val), callbacks=[WandbCallback()])
