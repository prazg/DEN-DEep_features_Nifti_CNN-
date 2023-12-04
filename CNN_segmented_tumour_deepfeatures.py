# CopyRight Prajwal Ghimire https://github.com/prazg/Deep-Features-Nifti-Network-DEN

import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

# Define the CNN model architecture
model = Sequential()

# First convolutional layer
model.add(Conv2D(96, (9, 9), strides=(4, 4), padding='valid', input_shape=(224, 224, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Second convolutional layer
model.add(Conv2D(256, (7, 7), strides=(1, 1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Third convolutional layer
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))

# Fourth convolutional layer
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))

# Fifth convolutional layer
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Flatten the output of the last pooling layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(4096))
model.add(Activation('relu'))  # Add activation function

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Define the directory that contains the patient folders
base_dir = r"C:\Users\Folderpath"  # Replace with the actual path

# Loop over each patient ID folder
for folder_name in os.listdir(base_dir):
    patient_dir = os.path.join(base_dir, folder_name)

    # Check if it's a directory
    if os.path.isdir(patient_dir):
        # Extract the patient ID from the folder name and drop '_nifti'
        patient_id = folder_name.replace('_nifti', '')

        # Load the segmented tumor file
        tumor_file = os.path.join(patient_dir, f'{patient_id}_tumor_segmentation.nii.gz')
        tumor_data = nib.load(tumor_file).get_fdata()

        # Preprocess the tumor data to match the input requirements of the CNN model
        # Resize the image to the expected input size of the model (224, 224 in this case)
        tumor_data_resized = np.array([image.array_to_img(slice_2d[..., None], scale=False).resize((224, 224)) for slice_2d in tumor_data])
        # Normalize the pixel values to 0-1 range
        tumor_data_normalized = tumor_data_resized / np.max(tumor_data_resized)
        # Expand the dimensions of the image array if the model expects a 4D input (batch size, height, width, channels)
        tumor_data_expanded = np.expand_dims(tumor_data_normalized, axis=-1)

        # Pass the preprocessed data through the CNN model to extract deep features
        features = model.predict(tumor_data_expanded)

import matplotlib.pyplot as plt

# Reshape the features into a 2D array for visualization
# Calculate the correct dimensions
dim1 = 64  # This can be any number that divides 983040 evenly
dim2 = features.size // dim1

# Reshape the features into a 2D array for visualization
features_reshaped = features.reshape((dim1, dim2))

# Use Matplotlib to display the image
plt.imshow(features_reshaped, cmap='viridis')
plt.colorbar()
plt.show()
# The number of features may vary depending on your model architecture
# Here we assume the features are of shape (1, 4096)


#save the feature file # Define the file path where you want to save the features
features_file_path = r"C:\Users\features.npy"

# Save the features to the file
np.save(features_file_path, features)

