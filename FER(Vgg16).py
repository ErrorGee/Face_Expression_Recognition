#import all the libraries
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten,MaxPool2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy,kullback_leibler_divergence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
import cv2 as cv
import numpy as np
import os
import sys
import shutil
import tensorflow as tf

os.listdir()

#load the dataset
path="../input/mma-facial-expression/MMAFEDB"
labels=os.listdir(path+'/train')
trainp=r'../input/mma-facial-expression/MMAFEDB/train'

#print labels
labels

#image preprocessing
test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
       height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, 
       channel_shift_range=10., horizontal_flip=True)
                               
train_batches = test_datagen.flow_from_directory(path+'/train', target_size=(224,224), classes=labels, batch_size=32)
valid_batches = ImageDataGenerator().flow_from_directory(path+'/valid', target_size=(224,224), classes=labels, batch_size=16)
test_batches = ImageDataGenerator().flow_from_directory(path+'/test', target_size=(224,224), classes=labels, batch_size=10)

#image view
imgs, labels = next(train_batches)

def plots(ims, figsize=(12,6), rows=4, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
print(plots(imgs, titles=labels))

#load the vgg model
vmodel = tf.keras.applications.vgg16.VGG16()


#see summary for vgg model
vmodel.summary()

#load the sequential model structure
model=Sequential()

#we are not using all of the layers from vgg original model, so we will remove some and add rest in our model
for i in vmodel.layers[:-1]:
    model.add(i)
model.summary()

#add the dense layer at last
model.add(Dense(7,activation='softmax'))

#compile the model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#fit the model with training and validation set
model.fit(train_batches, 
                    steps_per_epoch=250, 
                    validation_data=valid_batches, 
                    validation_steps=85,
                    epochs=50,callbacks=[checkpoint])
if os.path.isfile("FR_seq_model_4_layer/improved.h5") is False:
    final_model.save("FR_seq_model_4_layer/improved.h5")
os.listdir()
