import sys
import os
import cv2
import datetime
# import imutils
import numpy as np
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import pickle


data = []
labels = []
bboxes = []
imagePaths = []

# classes = ["trafficlight", "speedlimit", "crosswalk", "stop"]

annot_dir  = "./annotations"
images_dir = "./images"

for filename in os.listdir(annot_dir):
    f = os.path.join(annot_dir, filename)
    tree = ET.parse(f)
    root = tree.getroot()
    
    w = int(root.find('./size/width').text)
    h = int(root.find('./size/height').text)
    
    for box in root.findall('./bndbox'):
        xmin = int(box.find('xmin').text) / w
        ymin = int(box.find('ymin').text) / h
        xmax = int(box.find('xmax').text) / w
        ymax = int(box.find('ymax').text) / h
    
    label = root.find('./object/name').text
    
    imname = root.find('./filename').text
    impath = os.path.join(images_dir, imname)
    image = load_img(impath, target_size=(224,224))
    image = img_to_array(image)
    
    data.append(image)
    labels.append(label)
    bboxes.append((xmin,ymin,xmax,ymax))
    imagePaths.append(impath)

# normalize -> from [0-255] to [0-1]
data = np.array(data, dtype="float32") / 255.0

# convert to np arrays
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# test-train split 20%,80%

split = train_test_split(data,
                         labels,
                         bboxes,
                         imagePaths,
                         test_size=0.20,
                         random_state=12)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths,  testPaths)  = split[6:]

# saving test files for later use
with open("testing_multiclass.txt", "w") as f:
    f.write("\n".join(testPaths))

# The Neural Net : Architecture 
vgg = VGG16(weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

# freeze training any of the layers of VGGNet
vgg.trainable = False

# max-pooling is output of VGG, flattening it further
flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
# 4 neurons correspond to 4 co-ords in output bbox

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(
    inputs=vgg.input,
    outputs=(bboxHead, softmaxHead))

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 16

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}

testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

opt = Adam(INIT_LR)

model.compile(loss=losses, 
              optimizer=opt, 
              metrics=["accuracy"], 
              loss_weights=lossWeights)

#print(model.summary())


# Training
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1)

model.save("model_bbox_regression_and_classification", save_format="h5")

f = open("lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()



# Testing the model
def predict(imagePath):
     # loading input image
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # predicting bbox and label
    model = load_model("./model_bbox_regression_and_classification")
    lb = pickle.loads(open("./lb.pickle", "rb").read())
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    # finding class label with highest pred. probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]
    return label 
def main():
    path = "testing_multiclass.txt"
    filenames = open(path).read().strip().split("\n")
    for f in filenames:
        lablePredict=predict(f)
        print(lablePredict)


if __name__=="__main_":
    main()