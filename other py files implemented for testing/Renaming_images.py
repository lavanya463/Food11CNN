import os
import numpy as np
from keras.utils import np_utils
from sklearn.datasets import load_files

def load_dataset(path):
    data = load_files(path)
    food_files = np.array(data['filenames'])
    food_targets = np_utils.to_categorical(np.array(data['target']), 11)
    return food_files, food_targets


x_valid, y_valid = load_dataset('food11re_noval/validation')
count = 0
for i in x_valid:
    print(i[0:32])

for i in x_valid:
    # path should be given otherwise saving in file folder and creating a mess
    a = (i)[0:31]+"valid"+str(count)+".jpg"
    #print(i)
    #print(a)
    os.rename(i, a)
    count = count + 1

print("done")
