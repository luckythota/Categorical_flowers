# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:54:05 2023

@author: NAGA LAKSHMI
"""

import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

#directory with daise pictures
daisy_dir = os.path.join(r'F:\dl\flowers\daisy')

# Directory with dandelion pictures 
dandelion_dir = os.path.join(r'F:\dl\flowers\dandelion')

# Directory with rose pictures
rose_dir = os.path.join(r'F:\dl\flowers\rose')

# Directory with sunflower pictures
sunflower_dir = os.path.join(r'F:\dl\flowers\sunflower')

# Directory with tulip pictures
tulip_dir = os.path.join(r'F:\dl\flowers\tulip')
train_daisy_names = os.listdir(daisy_dir)
print(train_daisy_names[:5])

train_rose_names = os.listdir(rose_dir)
print(train_rose_names[:5])

batch_size = 128



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'F:\dl\flowers',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['daisy','dandelion','rose','sunflower','tulip'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(48,48)

#$input_shape = tuple(list(target_size)+[3])
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='sigmoid', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='sigmoid'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='sigmoid'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='sigmoid'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()


# Optimizer and compilation
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=0.01),metrics=['acc'])#RMSprop(lr=0.001)
# Total sample count  #adam(lr=0.1)
total_sample=train_generator.n
# Training
num_epochs = 10
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)



