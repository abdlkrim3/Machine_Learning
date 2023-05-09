import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the model and label binarizer
model = load_model('model_bbox_regression_and_classification')
lb = pickle.loads(open('lb.pickle', "rb").read())

# Define the function to predict class
def predict_class(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the class
    (bboxPreds, labelPreds) = model.predict(image)
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    # Return the predicted class
    return label

# Call the predict_class function
image_path = "./images/road94.png"
predicted_class = predict_class(image_path)

# Print the predicted class
print(predicted_class)