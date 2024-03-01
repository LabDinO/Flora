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
config.epochs = 10
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

# Example usage
# Specify the paths to the train and test folders
train_folder = 'D:/SMALL_IMAGES_CNN/train'
test_folder = 'D:/SMALL_IMAGES_CNN/test'

# Load and process training images
x_train_class1, y_train_class1 = load_and_process_images(f'{train_folder}/positive', label=1)
x_train_class2, y_train_class2 = load_and_process_images(f'{train_folder}/negative', label=0)

# Load and process testing images
x_test_class1, y_test_class1 = load_and_process_images(f'{test_folder}/positive', label=1)
x_test_class2, y_test_class2 = load_and_process_images(f'{test_folder}/negative', label=0)

# Concatenate the arrays for each class
x_train = np.concatenate([x_train_class1, x_train_class2], axis=0)
y_train_raw = np.concatenate([y_train_class1, y_train_class2], axis=0)
x_test = np.concatenate([x_test_class1, x_test_class2], axis=0)
y_test_raw = np.concatenate([y_test_class1, y_test_class2], axis=0)

# Assuming you have class names as a list
class_names = ['negative','positive']


##########################################################################
##########################################################################
##########################################################################

# Removing the last dimention, that is only 255 values
#I don't know why is that


##AQUI VOU FAZER ELE PEGAR SÓ A PRIMEIRA DIMENÇÃO JÁ QUE OS PIXELS SÃO IGUAIS.
x_test = x_test[:, :, :, 0]
x_train = x_train[:, :, :, 0]

#Image pixels
img_width = x_train.shape[1]
img_height = x_train.shape[2]


# One hot encode ouput
y_train = to_categorical(y_train_raw)
y_test = to_categorical(y_test_raw)
labels = range(1)
#Just shape
num_classes = y_train.shape[1]

#x_train = x_train.astype('float32')

#x_test = x_test.astype('float32')

# We build an extremely simple perceptron to try to fit our data

x_train_normalized = x_train / 255.
x_test_normalized = x_test / 255.



# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(config.hidden_nodes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=config.optimizer,
                    metrics=['accuracy'])

# Fit the model
model.fit(x_train_normalized, y_train, validation_data=(x_test_normalized, y_test), 
      epochs=config.epochs,
      callbacks=[WandbCallback(data_type="image", labels=labels)])