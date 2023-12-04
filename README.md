# Deep Features Nifti Network (DEN)
This repository defines a Convolutional Neural Network (CNN) model that extracts deep features from Nifti MRI files of segmented glioblastoma tumours
![DENnetwork](https://github.com/prazg/Deep-Features-Nifti-Network-DEN/assets/107046434/960b9016-a6d6-45d0-820d-73375297e4d8)

Here are the key characteristics of this CNN model:
1.	Architecture Overview:
•	The model is a sequential model, meaning it has a linear stack of layers.
•	It consists of five convolutional layers, followed by a flattening operation and a fully connected layer.

3.	Convolutional Layers:
•	First Layer: It has 96 filters, each of size 9×99×9, with a stride of 4×44×4 and 'valid' padding. This is followed by ReLU activation, batch normalization, and max pooling with a 3×33×3 window and 2×22×2 stride.
•	Second Layer: It has 256 filters of size 7×77×7, stride 1×11×1, and 'same' padding, followed by ReLU activation, batch normalization, and max pooling.
•	Third and Fourth Layers: Each has 384 filters of size 3×33×3 with a stride of 1×11×1 and 'same' padding, followed by ReLU activation.
•	Fifth Layer: It has 256 filters, similar to the third and fourth layers, and is followed by ReLU activation and max pooling.

4.	Fully Connected Layer:
•	After the convolutional and pooling layers, the model flattens the output and feeds it into a dense layer with 4096 units, followed by ReLU activation.

5.	Model Compilation:
•	The model is compiled with the Adam optimizer and uses categorical cross-entropy as the loss function.

6.	Data Preprocessing for Medical Imaging:
•	The code includes steps for loading and preprocessing medical images (NIfTI format). It involves resizing, normalizing, and expanding dimensions to match the input shape required by the CNN model.

7.	Feature Extraction and Visualization:
•	The model is used to predict features from preprocessed medical images. These features are then reshaped and visualized using a heatmap.

8.	Saving Extracted Features:
•	The extracted features are saved as a NumPy array to a specified file path.

9.	Input Shape:
•	The model expects an input shape of 224×224×1224×224×1, suitable for grayscale images of size 224x224 pixels.

Inspiration: https://www.mdpi.com/2075-4418/12/4/1018 (Basic Architecture was inpired by this paper: they donot provide any code for the network)


Copyright: Prajwal Ghimire
