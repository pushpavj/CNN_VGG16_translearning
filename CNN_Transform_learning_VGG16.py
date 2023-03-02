import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import pandas as pd
import seaborn as ssn
from pathlib import Path
import urllib.request as request

#****************************************************************************
#This module is used to download the and and bees image data set from the
#web site. This data set is used to train the neural network using the VGG16
# CNN architecture transfer learning algorithm.

#****************************************************************************

IMAGE_SIZE = (224, 224)
model = tf.keras.applications.VGG16(weights='imagenet',
                     include_top=True, #means you want to include the 
                     # fully connected layer as well other wise it will include
                     # till the convolution layer
                    input_shape=None, 
                    classes=1000 #pre trained model has 1000 classes

                     ) 

print("model summary", model.summary())

#save the model
model.save('CNN_VGG16_full_model.h5')
#check whta is the padding used
print("model padding", model.layers[1].padding)

#download the data
data_url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
def create_dir(dir_path):
    os.makedirs(dir_path,exist_ok=True)
    print(f"Directory {dir_path} created")
        
ROOT_DATA_DIR="data"
create_dir(ROOT_DATA_DIR)

data_zip_file="data.zip"
data_zip_path=os.path.join(ROOT_DATA_DIR,data_zip_file)

if not os.path.isfile(data_zip_path):
    print("downloading data")
    filename,headers = request.urlretrieve(data_url, data_zip_path)
    print(f"filename: {filename} created with info: {headers}")
else:
    print ("data already downloaded")
    print()

#unzip the data
from zipfile import ZipFile

def unzip_data(source:str,dest:str):
    with ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(dest)
unzip_data(data_zip_path,ROOT_DATA_DIR)

main_data_dir=Path(r"D:\user\jupyternotes\Praketh\CV-Computer vision\CNN_VGG16_translearning\data\hymenoptera_data\train")
#Path is a library that allows you to create and manipulate file paths.
# This will create standard directory path which is readable by the Os. 
# This path contains the path of the training data set.
print("main_data_dir",main_data_dir)

BATCH_SIZE=32
PIXCEL_PER_IMAGE=224
IMAGE_SIZE=(PIXCEL_PER_IMAGE,PIXCEL_PER_IMAGE)

#create data generators like data loaders in pytorch
dict()
datagen_keyword = dict(rescale=1./255,validation_split=0.20) # data generator
     #key workd arguments for data generator
#the parameter rescale is to multiply every pixel in the preprocessing 
# image. rescale: rescaling factor.
dataflow_kwargs=dict(target_size=IMAGE_SIZE,
                     batch_size=BATCH_SIZE,
                     )#data flow keyword arguments
valid_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
                            **datagen_keyword) #This will create a valid data generator

# The ImageDataGenerator class is used to generate batches of data.
#Generate batches of tensor image data with real-time data augmentation.
valid_generator=valid_datagen.flow_from_directory(
                                main_data_dir,
                                subset='validation',
                                shuffle=False,
                                **dataflow_kwargs)
#The flow_from_directory() method takes a path of a directory 
# and generates batches of augmented data. 
data_agumentation=False 

if data_agumentation:
    train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
                        rotation_range =40,
                        horizantal_flip =True,
                        width_shift_range =0.2,
                        height_shift_range =0.2,
                        shear_range =0.2,
                        zoom_range =0.2,
                        **datagen_keyword)
else:
    train_datagen = valid_datagen
    
train_generator=train_datagen.flow_from_directory(
                main_data_dir,subset="training",shuffle = True,
                **dataflow_kwargs)
RGB_IMAGE_SIZE=(PIXCEL_PER_IMAGE,PIXCEL_PER_IMAGE,3)

vgg= tf.keras.applications.vgg16.VGG16(
    input_shape=RGB_IMAGE_SIZE,
    weights='imagenet',
    include_top=False, #means it will not include the fully connected layer,
    # it will include till the convolution layer


)

print("vgg summary", vgg.summary())

for layer in vgg.layers:
    print(f"layer name: {layer.name} is trainable: {layer.trainable}")
    layer.trainable=False
#freeze all the layers



for layer in vgg.layers:
    print(f"layer name: {layer.name} is trainable: {layer.trainable}")

print("vgg.output",vgg.output)

CLASSES=2
x=tf.keras.layers.Flatten()(vgg.output)
predictions=tf.keras.layers.Dense(CLASSES,activation='softmax')(x)

new_model=tf.keras.models.Model(inputs=vgg.input,outputs=predictions)
print("new_model summary",new_model.summary())

new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) 

EPOCHS=10
history=new_model.fit(train_generator,
                                epochs=EPOCHS,
                                validation_data=valid_generator,
)         
                
new_model.save('CNN_VGG16_new_full_model.h5')

print("train_generator.class_indices",train_generator.class_indices)

label_map={val: key for key, val in train_generator.class_indices.items()}

print("label_map",label_map)

#take one sample image from the validation ant library
test_image=plt.imread(Path(r"D:\user\jupyternotes\Praketh\CV-Computer vision\CNN_VGG16_translearning\data\hymenoptera_data\val\ants\17081114_79b9a27724.jpg"))

plt.imshow(test_image)

plt.show()

#new_model.predict(test_image)

print("test_image.shape",test_image.shape)

#need to reshape the image to be able to use it in the model

resized_img=tf.image.resize(test_image,(224,224))

#need to expand the image to be able to use it in the model
input_image=tf.expand_dims(resized_img,axis=0)

pred=new_model(input_image)
print("pred.shape",pred.shape)
print("pred",pred)

pred_index=np.argmax(pred,axis=1)
print("predicted value",label_map[pred_index[0]])

loaded_model=tf.keras.models.load_model('CNN_VGG16_new_full_model.h5')
preprocessed_image=tf.keras.applications.vgg16.preprocess_input(test_image)

print("preprocessed_image.shape",preprocessed_image.shape)

plt.imshow(preprocessed_image[0])
plt.show()


