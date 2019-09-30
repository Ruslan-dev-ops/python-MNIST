# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:44:55 2019

@author: Ruslan
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# input image dimensions
img_rows, img_cols = 28, 28
classes=[0,1,2,3,4,5,6,7,8,9]

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

try:#подгрузка созданных весов
    file = open('res//weights.hdf')
except IOError:
    print('weights not found')
else:
    file.close()  
    model.load_weights("res//weights.hdf")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

y_pred=model.predict_classes(x_test)
con_mat = confusion_matrix(y_test, y_pred)#расчёт матрицы
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

figure = plt.figure(figsize=(8, 8))#отрисовка матрицы
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()