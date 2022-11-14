import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import cv2
import warnings
#warnings.filterwarnings("ignore")
# IMPORT KERAS LIBRARY
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.metrics import AUC
from scipy.ndimage import gaussian_filter
##########################

def build_model(PATH,image_size = 224, load_previous_weights = True, freeze_cnn = False):
    base_model = DenseNet121(include_top= False, input_shape=(image_size,image_size,3), weights='imagenet')
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D(input_shape=(1024,1,1))(x)
    # Add a flattern layer 
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # Add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # and a logistic layer --  we have 5 classes
    predictions = Dense(6, activation='sigmoid')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Recover previously trained weights
    if load_previous_weights:
        try:
            model.load_weights(PATH)
            print('Weights successfuly loaded')
        except:
            print('Weights not loaded')
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if freeze_cnn:
        for layer in base_model.layers:
            layer.trainable = False     
    # compile the model (should be done *after* setting layers to non-trainable)
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', auc])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', AUC()])
    return model

def pre_process(im_1):
  chexpert_targets=['None            ', 'Atelectasis     ', 'Cardiomegaly    ', 'Consolidation   ', 'Edema           ', 'Pleural Effusion']
  #im_1=a=cv2.imread(img_path)
  im_2=cv2.resize(im_1,(224,224))
  im_3=im_2.reshape(im_2.shape + (1,))
  im_4=im_3.transpose(3,0,1,2)
  img_process=im_4/255
  return chexpert_targets,img_process

def heater(cnn,image_in_4d):
  with tf.GradientTape() as tape:
    last_conv_layer = cnn.get_layer('conv5_block16_concat')
    iterate = tf.keras.models.Model([cnn.inputs], [cnn.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(image_in_4d)
    class_out = model_out[:, np.argmax(model_out[0])]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  hm=np.squeeze(heatmap)
  return hm

def predecir(iamgen):
    labels,img_prep=pre_process(iamgen)
    prediction = cnn.predict(img_prep)
    heatmapX=heater(cnn,img_prep)
    resized = cv2.resize(heatmapX, (224,224), interpolation = cv2.INTER_AREA)
    heatmap1 = np.uint8(255 * resized)
    heatmap1 = gaussian_filter(heatmap1, sigma=11)
    heatmap2 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    img = np.uint8(255 * img_prep[0])
    superimposed_img = cv2.addWeighted(img, 0.8, heatmap2, 0.4, 0)
    return superimposed_img,prediction
######################################
## CARGA LA RED

image_size_input = 224
cnn = build_model('weights.hdf5',image_size = image_size_input)

## PREDICTION
