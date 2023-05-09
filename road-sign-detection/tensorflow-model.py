import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
#df = pd.read_csv('')
annot_dir  = "./annotations"
images_dir = "./images"

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create data generators for the training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, directory='my_data/train/',
    x_col="Filename", y_col="ClassId", target_size=(64, 64), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory='my_data/train/',
    x_col="Filename", y_col="ClassId", target_size=(64, 64), batch_size=32, class_mode='categorical')

# Build the CNN model
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Predict the class labels of the test set
preds = model.predict(test_generator)
predicted_classes = np.argmax(preds, axis=1)

# Get the true class labels of the test set
true_classes = test_generator.classes

# Calculate the test accuracy
acc = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"Test accuracy: {acc}")