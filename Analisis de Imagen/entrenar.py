from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16

import pandas as pd
from pandas import DataFrame, read_csv
import cv2, numpy as np
#import numpy as np
from shutil import copyfile
from PIL import Image

from sklearn.metrics import confusion_matrix
#####################################################################################################################
# DataTesis 100
# bdtesis2 150
image_size = 100
train_dir = 'C:/Users/asus/Desktop/proyecto/DATA100/DataTesis/train'#DataTesis/train'
validation_dir = 'C:/Users/asus/Desktop/proyecto/DATA100/DataTesis/validate'

#Load the VGG model, 
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
#for layer in vgg_conv.layers:
#    print(layer, layer.trainable)

from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam, Nadam


# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())#Flattens the input. Does not affect the batch size
model.add(layers.Dense(4096, activation='relu'))#capas full connected  con la funcion de activacion Rectified Linear Unit
model.add(layers.Dense(4096, activation='softmax'))
model.add(layers.Dense(2, activation='softmax'))

vgg_conv.summary()#muestra como es la estructura de la red neuronal

# No Data augmentation 
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 32# configurar el tamaño del batch para el entrenamiento como en el test
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])#List of metrics to be evaluated by the model during training and testing

print("training")

# Train the Model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=65,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)
# Save the Model
#model.save('C:/Users/asus/Desktop/proyecto/all_freezed.h5')
model.save('C:/Users/asus/Downloads/all_freezed.h5')
# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


################################

#vgg16_pred = model.predict(validation_generator, batch_size=32, verbose=1)  
#vgg16_predicted = np.argmax(vgg16_pred, axis=1)  

#print(vgg16_pred)

# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator muestra las clases a que pertence cada muestra
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors

for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],#nombre de la etiqueta
        pred_label,#predicciion
        predictions[errors[i]][pred_class])#porcentaje de prediccíon
    
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
