import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os
#***************************************************************************************************
# The function of this module is to TFlite model inference. And also optimized model inference.
# In this module, we will download the VGG16 model which was created in the
# CNN_Transform_learning_VGG16.py module.
# Then we will convert this VGG model to TFlite model.
# The benifit of converting to TFlite model is that the size of the model
# will reduce dastrically from 500MB to 50MB around.
# To further reduce the size of the model, we will build the optimized model
# via building the quantized model.
# The quantized model will conver the model into binary object.
# In this model we will also see how to use TFlite and quantized model
# to predict the class of an image.
#***************************************************************************************************
#Get the path of the VGG16 new model
path_of_vgg16_trained_model=Path(r"D:\user\jupyternotes\Praketh\CV-Computer vision\CNN_VGG16_translearning\CNN_VGG16_new_full_model.h5")
# Download the VGG16 new model

model=tf.keras.models.load_model(path_of_vgg16_trained_model)

#create the converted TFliet model
converter=tf.lite.TFLiteConverter.from_keras_model(model)
#this will create the converted model

tflite_model=converter.convert()
#this will create the tflite model

#save this tflite model under new directory tflite_model_dir

tflite_model_path=Path("./tflite_model_dir")
tflite_model_path.mkdir(parents=True, exist_ok=True) 
#create the directory if it doesn't exist

#save the tflite model

tfliet_model_file=tflite_model_path/"vgg16_model.tflite" 
tfliet_model_file.write_bytes(tflite_model)
#The tflite model is saved in the tflite_model_dir folder
#The tfliet model is created to reduce the size of the model

#To further reduce the size of the model, we need to convert the 
# tflite model to a binary model
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model=converter.convert()

#now we need to quantize the tflite model

tflite_model_quant_file=tflite_model_path/"vgg16_quant_model.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model) 
#quantized model is saved in the tflite_model_dir folder


#Verifying the quantized model
#Following stpes involved in running the quantized model or tensorflow lite model
#1. load the quantized model
#2. Build an interpreter for the quantized model
#3. Invoke the interpreter to get the input and output tensors
#4. Read output tensors from the interpreter

interpreter=tf.lite.Interpreter(model_path=str(tfliet_model_file))
#pass the model file path to the interpreter

interpreter.allocate_tensors()

interpreter_quant=tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()

#let us take smaple image for testing

test_image=plt.imread(r"D:\user\jupyternotes\Praketh\CV-Computer vision\CNN_VGG16_translearning\data\hymenoptera_data\train\bees\39747887_42df2855ee.jpg")
plt.imshow(test_image)
plt.show()
print("input details",interpreter.get_input_details())
print(interpreter.get_input_details()[0])

print(interpreter.get_input_details()[0]["shape"])

#To check what is the shape of the input tensor
#The shape of the input tensor is [  1 224 224   3]
#We need to change the shape of the test image to [1,224,224,3]
#print(tf.image.resize(test_image,(224,224)).shape)
resized_image=tf.image.resize(test_image,(224,224))

input_data=tf.expand_dims(resized_image,axis=0)
print("inputa_data.shape",input_data.shape)

input_index=interpreter.get_input_details()[0]["index"]
output_index=interpreter.get_output_details()[0]["index"]
print("output_details",interpreter.get_output_details())
#Out put details gives the details of the output shape that indicates the 
# number of prediction classes it will be producing.
print("output_index",output_index)
#pass the test image to the interpreter at its input index
interpreter.set_tensor(input_index,input_data)
interpreter.invoke()
#predict the class of the input image
pred= interpreter.get_tensor(output_index)
#gives the probability of the class of the input image
print("pred is",pred)

label_map={0:"ants",1:"bees"}
#get the predicted class by taking the argmax of the probability
argmax=tf.argmax(pred[0]).numpy()
print("argmax is",argmax)
print(label_map[argmax])
#predict the class of the input image using quantized model
input_index_quant=interpreter_quant.get_input_details()[0]["index"]
output_index_quant=interpreter_quant.get_output_details()[0]["index"]
#pass the test image to the interpreter at its input index
interpreter.set_tensor(input_index_quant,input_data)
interpreter.invoke()

pred_quant= interpreter.get_tensor(output_index_quant)
#gives the probability of the class of the input image
print("pred is",pred)
print("pred is",pred_quant)
#get the predicted class by taking the argmax of the probability
argmax_quant=tf.argmax(pred_quant[0]).numpy()
print(label_map[argmax_quant])






