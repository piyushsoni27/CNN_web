#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:03:26 2019

@author: piyush
"""

import os
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import cv2
import matplotlib.pyplot as plt
import numpy as np


filename = "3.jpeg"

# The names of the classes in the dataset.
CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

model_path = "/media/piyush/New Volume/Projects/CNN_web/model/architecure.h5"
model = keras.models.load_model(model_path)

model.compile(loss='categorical_crossentropy',
             optimizer=SGD(
                  lr=0.001, 
                  momentum=0.9),
              metrics=['accuracy'])

test_im_path = "/media/piyush/New Volume/Projects/CNN_web/test_images/"

print(os.path.join(test_im_path, filename))

img = cv2.imread(os.path.join(test_im_path, filename))
print(img.shape)
img = cv2.resize(img,(75,75))
img = np.reshape(img,[1,75,75,3])

img = img/255.

plt.axis('off')
plt.imshow(img[0])
plt.show()

prediction = model.predict(img)

print(CLASS_NAMES[np.argmax(prediction)])
