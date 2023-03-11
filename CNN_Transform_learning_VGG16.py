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
#In this module first we download the VGG16 model which is pretrained model for 1000 classifications.
#Then we modify this VGG16 model to be able to classify the bees images as 2 number of classifications.
#Then we will train the neural network using the modified VGG16 model.
#Then we will save the trained neural network to a file.
#The size of the model saved will be of huge i.e. 500 mb
# So in order to reduce the size of the model we need to implement TFLite which donwloads the above
#saved model and converts it to a.tflite file. The size of the .tflite will be of reduced one around 50MB
#Further we can reduce the size of the .tflite model by using the model optimizer by convertint the
#tflite model to a binary object model via quantized model.
#For this we have written TFlite.py module
#****************************************************************************

IMAGE_SIZE = (224, 224)
model = tf.keras.applications.VGG16(weights='imagenet',
                     include_top=True, #means you want to include the 
                     # fully connected layer as well other wise it will include
                     # till the convolution layer
                    input_shape=None, 
                    classes=1000 #pre trained model has 1000 classes

                     ) #This will download the VGG16 model which is pretrained model 
                        #for 1000 classifications


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
        
ROOT_DATA_DIR="data" #directory where the data will be downloaded

create_dir(ROOT_DATA_DIR)

data_zip_file="data.zip" #name of the zip file where the data will be downloaded

data_zip_path=os.path.join(ROOT_DATA_DIR,data_zip_file)

if not os.path.isfile(data_zip_path): #check if the data zip file is already downloaded
    print("downloading data")
    filename,headers = request.urlretrieve(data_url, data_zip_path)
    print(f"filename: {filename} created with info: {headers}")
else:
    print ("data already downloaded")
  


#unzip the data
from zipfile import ZipFile

def unzip_data(source:str,dest:str):
    with ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(dest)
unzip_data(data_zip_path,ROOT_DATA_DIR) #unzip the data to the Root directory


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
     #key word arguments for data generator
#the parameter rescale is to multiply every pixel in the preprocessing to resize the image
# image. rescale: rescaling factor.
dataflow_kwargs=dict(target_size=IMAGE_SIZE,
                     batch_size=BATCH_SIZE,
                     )#data flow keyword arguments to the data generator with the target_size 
                      #with the given batch_size


valid_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
                            **datagen_keyword) #This will create a valid data generator

# The ImageDataGenerator class is used to generate batches of data.
#Generate batches of tensor image data with real-time data augmentation.
valid_generator=valid_datagen.flow_from_directory(
                                main_data_dir,
                                subset='validation',
                                shuffle=False,
                                **dataflow_kwargs)
#The flow_from_directory() method takes a path of a train data directory 
# and generates batches of augmented data. 
data_agumentation=False 

#perform data augmentation if the data is a training data set

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
                **dataflow_kwargs) #this will generate batches of augmented data

RGB_IMAGE_SIZE=(PIXCEL_PER_IMAGE,PIXCEL_PER_IMAGE,3)

#download the VGG16 model till the convolution layer for the required image_shape or size
vgg= tf.keras.applications.vgg16.VGG16(
    input_shape=RGB_IMAGE_SIZE,
    weights='imagenet',
    include_top=False, #means it will not include the fully connected layer,
    # it will include till the convolution layer


)

print("vgg summary", vgg.summary())

#freeze all the layers
for layer in vgg.layers:
    print(f"layer name: {layer.name} is trainable: {layer.trainable}")
    layer.trainable=False # freeze all the layers by setting trainable as False


#check if the layers are frozen or not

for layer in vgg.layers:
    print(f"layer name: {layer.name} is trainable: {layer.trainable}")

print("vgg.output",vgg.output) # this will print the fully connected layer


CLASSES=2 #set the number of classes for the model

x=tf.keras.layers.Flatten()(vgg.output)# This will flatten the fully connected layer

predictions=tf.keras.layers.Dense(CLASSES,activation='softmax')(x) #This will create the output layer

print("vgg.output",vgg.output)
print("predictions",predictions)

new_model=tf.keras.models.Model(inputs=vgg.input,outputs=predictions) # this will create a new model
                        #based on the required input image size and number of classes

print("new_model summary",new_model.summary())

new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) 

EPOCHS=10 
#for the number of epoches the training data and validation data will be generated for the 
# given number of batches and the model will be trained for the given number of epochs

history=new_model.fit(train_generator,
                                epochs=EPOCHS,
                                validation_data=valid_generator,
)         
                
new_model.save('CNN_VGG16_new_full_model.h5') #save the new model


print("train_generator.class_indices",train_generator.class_indices) #interchange class name and 
                                                                    # values


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
input_image=tf.expand_dims(resized_img,axis=0) #expand will add 1 diminsion at the beginning to the
                                               #image shape

pred=new_model(input_image) #predict the given image.i.e. provides the probability of each output
                            # class here it is 2 classes ants and bees
print("pred.shape",pred.shape)
print("pred",pred) 

pred_index=np.argmax(pred,axis=1) #select the index of the highest probability

print("predicted value",label_map[pred_index[0]])

#download the saved new model
loaded_model=tf.keras.models.load_model('CNN_VGG16_new_full_model.h5')
preprocessed_image=tf.keras.applications.vgg16.preprocess_input(test_image) #preprocess_input is a
#function that takes an image and returns a preprocessed image


print("preprocessed_image.shape",preprocessed_image.shape)

plt.imshow(preprocessed_image[0])
plt.show()


