from tensorflow import keras
import numpy as np
from keras.applications.mobilenet import preprocess_input
from keras_preprocessing.image import load_img
from keras_preprocessing import image
import os

from tensorflow import keras
model = keras.models.load_model("VGG16Adam.h5")

def predict (model , img ):
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]



text_file1 = open("falseNormalResNetADAM.txt","w")
img_path="./chest_xray/test/NORMAL"

vrai = 0
total = 0
for idx ,img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp','jpeg','jpg','png','tif','tiff')):
        continue
    print(img_name)
    filepath=os.path.join(img_path,img_name)
    img = load_img(filepath,target_size=(64,64))
    total+=1
    preds = predict(model,img)
    if preds[0]>=0.5:
        vrai+=1
    else :
        text_file1.write(filepath+"\n")
    print(preds)
acc1 = (vrai/total)*100
text_file1.close()




text_file2 = open("falsePneumResNetADAM.txt","w")
img_path="./chest_xray/test/PNEUMONIA"

vrai = 0
total = 0
for idx ,img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp','jpeg','jpg','png','tif','tiff')):
        continue
    print(img_name)
    filepath=os.path.join(img_path,img_name)
    img = load_img(filepath,target_size=(64,64))
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