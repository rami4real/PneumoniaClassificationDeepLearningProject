import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import plot_model
import pickle
import cv2
from sklearn.utils import shuffle
def show_history():

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.legend()

    plt.show()


# 0.003 -> 0.005
output_size = (128,128)
dirname = "chest_xray/train/"
index1 = 0.003
def get_data(dirname,index1,output_size):
    x_data = []
    y_data = []
    index2 = index1 + 0.001
    index3 = index2 + 0.001
    for folder in os.listdir(dirname):
        linkFolder = os.path.join(dirname,folder)
        print(linkFolder)
        for file in os.listdir(linkFolder):
            linkFile = os.path.join(linkFolder,file)
            data_img = cv2.imread(linkFile)
            data_img = cv2.resize(data_img,output_size)
            x_data.append(np.array(data_img))
            y_data.append(linkFolder.split('/')[1])
    return x_data,y_data

x_data = []
y_data = []
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
dirname = "chest_xray/train/"
x_train,y_train = get_data(dirname,index1,output_size)
print(np.array(x_train).shape)
print(np.array(y_train).shape)

dirname = "chest_xray/test/"
x_test,y_test = get_data(dirname,index1,output_size)
print(np.array(x_test).shape)
print(np.array(y_test).shape)

dirname = "chest_xray/val/"
x_val,y_val = get_data(dirname,index1,output_size)
print(np.array(x_val).shape)
print(np.array(y_val).shape)

encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)

x_train, y_train = shuffle(x_train, y_train, random_state=5)
x_test, y_test = shuffle(x_test, y_test, random_state=5)

x_train, x_val,y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val)
input_shape = (x_train[1].shape)
input_shape

import tensorflow as tf

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)


model = Sequential()
model.add(ResNet50_model)


model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))


model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
model.summary()

history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val), shuffle=False, callbacks=[model_checkpoint])
show_history()

import numpy as np
from keras.applications.mobilenet import preprocess_input
from keras_preprocessing.image import load_img
from keras_preprocessing import image

def predict (model , img ):
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]



text_file1 = open("falseDetectionNormal.txt","w")
img_path="./chest_xray/test/NORMAL"

vrai = 0
total = 0
for idx ,img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp','jpeg','jpg','png','tif','tiff')):
        continue
    print(img_name)
    filepath=os.path.join(img_path,img_name)
    img = load_img(filepath,target_size=(128,128))
    total+=1
    preds = predict(model,img)
    if preds[0]>=0.5:
        vrai+=1
    else :
        text_file1.write(filepath+"\n")
    print(preds)
acc1 = (vrai/total)*100
text_file1.close()




text_file2 = open("falseDetectionPneumonia.txt","w")
img_path="./chest_xray/test/PNEUMONIA"

vrai = 0
total = 0
for idx ,img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp','jpeg','jpg','png','tif','tiff')):
        continue
    print(img_name)
    filepath=os.path.join(img_path,img_name)
    img = load_img(filepath,target_size=(128,128))
    total+=1
    preds = predict(model,img)
    if preds[1]>=0.5:
        vrai+=1
    else :
        text_file2.write(filepath+"\n")
    print(preds)
acc2 = (vrai/total)*100
print("acc Normal : "+str(acc1))
print("acc Pneum : "+str(acc2))
text_file2.close()