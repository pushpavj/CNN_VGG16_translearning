import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os

path_of_vgg16_trained_model=Path(r"D:\user\jupyternotes\Praketh\CV-Computer vision\CNN_VGG16_translearning\CNN_VGG16_new_full_model.h5")
model=tf.keras.models.load_model(path_of_vgg16_trained_model)

#create the coverted model
converter=tf.lite.TFLiteConverter.from_keras_model(model)
#this will create the converted model

tflite_model=converter.convert()
#this will create the tflite model

#save this tflite model

tflite_model_path=Path("./tflite_model_dir")
tflite_model_path.mkdir(parents=True, exist_ok=True) #create the directory if it doesn't exist

tfliet_model_file=tflite_model_path/"vgg16_model.tflite"
tfliet_model_file.write_bytes(tflite_model)
#The tflite model is saved in the tflite_model_dir folder
#The tfliet model is created to reduce the size of the model

#To further reduce the size of the model, we need to convert the tflite model to a binary model
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model=converter.convert()

#now we need to quantize the tflite model

tflite_model_quant_file=tflite_model_path/"vgg16_quant_model.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model) #quantized model is saved in the tflite_model_dir folder


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

#print(tf.image.resize(test_image,(224,224)).shape)
resized_image=tf.image.resize(test_image,(224,224))

input_data=tf.expand_dims(resized_image,axis=0)
print("inputa_data.shape",input_data.shape)

input_index=interpreter.get_input_details()[0]["index"]
output_index=interpreter.get_output_details()[0]["index"]
print("output_details",interpreter.get_output_details())

print("output_index",output_index)

interpreter.set_tensor(input_index,input_data)
interpreter.invoke()
pred= interpreter.get_tensor(output_index)
print("pred is",pred)

label_map={0:"ants",1:"bees"}

argmax=tf.argmax(pred[0]).numpy()
print("argmax is",argmax)
print(label_map[argmax])

input_index_quant=interpreter_quant.get_input_details()[0]["index"]
output_index_quant=interpreter_quant.get_output_details()[0]["index"]

interpreter.set_tensor(input_index_quant,input_data)
interpreter.invoke()

pred_quant= interpreter.get_tensor(output_index_quant)
print("pred is",pred_quant)
argmax_quant=tf.argmax(pred_quant[0]).numpy()
print(label_map[argmax_quant])






