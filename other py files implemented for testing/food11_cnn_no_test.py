# import the necessary libraries
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
import random
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_files
from keras import backend as K
from keras.models import model_from_json
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing import image
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, GlobalAveragePooling2D



# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    food_files = np.array(data['filenames'])
    food_targets = np_utils.to_categorical(np.array(data['target']), 11)
    return food_files, food_targets


# pre processing the data to suit keras
def path_to_tensor(img_path, target_row, target_col, channels):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(target_row, target_col))
    # convert PIL.Image.Image type to 3D tensor with shape (target_size, target_size, channels)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, target_size, target_size, channels) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, row, col, channels):
    list_of_tensors = [path_to_tensor(img_path, row, col, channels) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def print_sizes(x_train, x_valid, x_test):
    # check sizes of dataset
    print('Number of Training images --> %d.' % len(x_train))
    print('Number of Validation images --> %d.' % len(x_valid))
    print('Number of Test images --> %d.' % len(x_test))


def shuffle_data(x_data, y_data):
    # to remove any bias, if present, shuffle the data. zip the contents so the category and the image will be packed together
    zip_data = list(zip(x_data, y_data))
    random.shuffle(zip_data)
    x_data[:], y_data[:] = zip(*zip_data)
    print(y_data)
    return x_data, y_data


def normalize(x_train, x_valid, x_test, imagesize_row, imagesize_col, channels):
    # pre-process the data for Keras (Normalize)
    x_train = paths_to_tensor(x_train, imagesize_row, imagesize_col, channels).astype('float32')/255
    x_valid = paths_to_tensor(x_valid, imagesize_row, imagesize_col, channels).astype('float32')/255
    x_test = paths_to_tensor(x_test, imagesize_row, imagesize_col, channels).astype('float32')/255

    return x_train, x_valid, x_test


def get_input_size(x_train, x_test, target_row, target_col, channels):
    # Find which format to use (depends on the backend), and compute input_shape
    img_rows, img_cols, channels = target_row, target_col, channels

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    return x_train, x_test, input_shape


def cnn_architecture(input_shape):
    # define the NN architecture
    # input_shape = (28,28,3)
    nn = Sequential()
    nn.add(Conv2D(64, 5, 5, activation='relu', input_shape=input_shape))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(32, 5, 5, activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(16, 5, 5, activation='relu'))
    nn.add(MaxPooling2D(pool_size=(3, 3)))
    nn.add(Flatten())
    # nn.add(GlobalAveragePooling2D())
    nn.add(Dense(700, activation='relu'))
    nn.add(Dropout(0.4))
    nn.add(Dense(600, activation='relu'))
    nn.add(Dense(11, activation='softmax'))
    nn.summary()

    return nn


def cnn_architecture_with_dropout(input_shape):
    model = Sequential()
    model.add(Conv2D(64, 3, 3, activation='relu', input_shape=input_shape))
    model.add(GlobalAveragePooling2D(pool_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, 3, 3, activation='relu', input_shape=input_shape))
    model.add(GlobalAveragePooling2D(pool_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dense(16, activation='softmax'))
    model.add(Dropout(0.2))
    model.add(Dense(11, activation='softmax'))

    return model



def save_cnn_architecture(model, filename):
    # model visualization
    # we can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
    plot_model(model, to_file=filename, show_shapes=True)


def accuracy_loss_plots(history):

    # accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('food11_cnn_accuracy.pdf')
    plt.close()
    # loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('food11_cnn_loss.pdf')


def plot_confusion_matrix(confusion_matrix, target_names):
    # when we pass a confusion matrix and the target names it produces a plot of the confusion matrix

    df_cm = pd.DataFrame(confusion_matrix, index=target_names, columns=target_names)
    print(df_cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig("confusion_matrix_plot.png")


def prediction_results(model, x_test, y_test):
    # compute probabilities
    pred_y = model.predict(x_test)
    # assign most probable label
    y_pred = np.argmax(pred_y, axis=1)
    # plot statistics
    print('Analysis of results')
    target_names = ['Bread', 'Dairy_product', 'Dessert', 'Egg', 'Fried_food', 'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Veggies']
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))


def save_weights(model, score):
    # saving model and weights
    model_json = model.to_json()
    with open('food.json', 'w') as json_file:
            json_file.write(model_json)
    weights_file = "weights-food11_"+str(score[1])+".hdf5"
    model.save_weights(weights_file, overwrite=True)


def load_model_weights(weights_file):
    # loading model and weights
    json_file = open('food.json', 'r')
    nn_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(nn_json)
    loaded_model.load_weights(weights_file)

    return loaded_model

from keras_preprocessing.image import ImageDataGenerator


def memory_managing():
    print('Using Keras version', keras.__version__)

    train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, width_shift_range=0.2, height_shift_range=0.2, channel_shift_range=30, rescale=1./255)
    valid_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, width_shift_range=0.2, height_shift_range=0.2, channel_shift_range=30,rescale=1./255)

    train_generator = train_datagen.flow_from_directory(directory="food11re_notest/training/", target_size=(200, 200), color_mode="rgb", batch_size=120, class_mode="categorical", shuffle=True, seed=25)
    valid_generator = valid_datagen.flow_from_directory(directory="food11re_notest/valid/", target_size=(200, 200), color_mode="rgb", batch_size=120, class_mode="categorical", shuffle=True, seed=25)

    input_size = (200, 200, 3)
    model = cnn_architecture(input_size)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    train_size = train_generator.n//train_generator.batch_size
    valid_size = valid_generator.n//valid_generator.batch_size

    history = model.fit_generator(generator=train_generator, steps_per_epoch=train_size, validation_data=valid_generator, validation_steps=valid_size, epochs=35)

    accuracy_loss_plots(history)

    score = model.evaluate_generator(generator=valid_generator, steps=len(valid_generator))

    print('valid loss:', score[0])
    print('valid accuracy:', score[1])

    Y_pred = model.predict_generator(valid_generator, steps=len(valid_generator), verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in y_pred]

    print('Confusion Matrix')
    confusion_m = confusion_matrix(valid_generator.classes, y_pred)
    print(confusion_m)
    print('Classification Report')
    target_names = ['Bread', 'Dairy_product', 'Dessert', 'Egg', 'Fried_food', 'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Veggies']
    print(classification_report(valid_generator.classes, y_pred, target_names=target_names))
    plot_confusion_matrix(confusion_m, target_names)

    print("finished")


if __name__ == "__main__":
    memory_managing()
