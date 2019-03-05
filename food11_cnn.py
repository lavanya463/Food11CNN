from __future__ import division
import keras
print( 'Using Keras version', keras.__version__)
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from PIL import ImageFile


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 11)
    return dog_files, dog_targets

#Load the food11 dataset
x_train, y_train = load_dataset('food11re/training')
x_valid, y_valid = load_dataset('food11re/validation')
x_test, y_test = load_dataset('food11re/evaluation')

# preprocessing the data to suit keras
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(28, 28))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

#Check sizes of dataset
print('Training dog images --> %d.' % len(x_train))
print('Validation dog images --> %d.' % len(x_valid))
print('Test dog images --> %d.' % len(x_test))

print( 'Number of train examples', x_train.shape[0])
print( 'Size of train examples', x_train.shape[1:])
#image_dim = x_train.shape[1]*x_train.shape[2]
#print(image_dim)

#Adapt the data as an input of a fully-connected (flatten to 1D)
#x_train = x_train.reshape(x_train.shape[0],(28*28))
#x_test = x_test.reshape(x_test.shape[0], (28*28))


# pre-process the data for Keras (Normalize)

x_train = paths_to_tensor(x_train[:9000]).astype('float32')/255
x_valid = paths_to_tensor(x_valid[:500]).astype('float32')/255
x_test = paths_to_tensor(x_test[:500]).astype('float32')/255



#Normalize data
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train = x_train / 255
#x_test = x_test / 255

#Adapt the labels to the one-hot vector syntax required by the softmax
from keras.utils import np_utils
#y_train = np_utils.to_categorical(y_train, 10)
#y_test = np_utils.to_categorical(y_test, 10)


#Find which format to use (depends on the backend), and compute input_shape
from keras import backend as K
#food11 resolution
img_rows, img_cols, channels = 28, 28, 3

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, 3)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers
input_shape = (28,28,3)
nn = Sequential()
nn.add(Conv2D(16, 5, 5, activation='relu', input_shape=input_shape))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(32, 5, 5, activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
#nn.add(Conv2D(64, 2, 2, activation='relu'))
#nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
#nn.add(GlobalAveragePooling2D())
# nn.add(Dense(16, activation='relu'))
nn.add(Dense(11, activation='softmax'))
nn.summary()


#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
from keras.utils import plot_model
plot_model(nn, to_file='nn.png', show_shapes=True)

#Compile the NN
nn.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#y_train = y_train.squeeze()
#y_valid = y_valid.squeeze()
#Start training
history = nn.fit(x_train, y_train[:9000],
          validation_data=(x_valid, y_valid[:500]),
          epochs=30, batch_size=120,verbose=1)

#Evaluate the model with test set
score = nn.evaluate(x_test, y_test[:500], verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('food11_cnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('food11_cnn_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = nn.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
print(classification_report(np.argmax(y_test[:500],axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test[:500],axis=1), y_pred))

#Saving model and weights
from keras.models import model_from_json
nn_json = nn.to_json()
with open('nn.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "weights-food11_"+str(score[1])+".hdf5"
nn.save_weights(weights_file,overwrite=True)

#Loading model and weights
json_file = open('nn.json','r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)
nn.load_weights(weights_file)
