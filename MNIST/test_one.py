# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:07:14 2019

@author: Ruslan
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

img_rows, img_cols = 28, 28
classes=[0,1,2,3,4,5,6,7,8,9]
name='test3_1.png'#название проверяемого фала
data=[]
inverse=True#инверсия значений пикселов

input_shape = (img_rows, img_cols, 1)

img = Image.open('res//'+name)
a = np.array(img)

plt.imshow(img, cmap = 'gist_gray')#отрисовка

if(a.shape[0]!=28|a.shape[1]!=28):#приведение размеров
    w, h = img.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h))
    if img_cols and img_rows:
        max_size = (img_cols, img_rows)
    elif img_cols:
        max_size = (img_cols, h)
    elif img_rows:
        max_size = (w, img_rows)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
    img.thumbnail(max_size, Image.ANTIALIAS)
    a = np.array(img)

img.close

data.append(a)

data = np.array(data)
data = data.reshape(1, img_rows, img_cols, 1)

data = data.astype('float32')
data /= 255
if(inverse):
    data -= 1
    data *= -1

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

pred=model.predict(data)

print('вероятно, это ',classes[pred[0].argmax(0)])