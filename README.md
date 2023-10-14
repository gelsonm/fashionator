# DeepFashion: Fashion Image Generation with Deep Learning
Welcome to the DeepFashion repository, where we explore the use of state-of-the-art deep learning techniques to generate high-fashion clothing and styles. This project involves the conversion of the DeepFashion dataset to COCO format, the implementation of object detection models, and the creation of high-quality fashion images using Generative Adversarial Networks (GANs) with PyTorch.

## File Descriptions

1. .yaml Files: These files store the configuration details of the object detection models used in our research.

2. DeepFashion2Coco.ipynb: This Jupyter notebook provides instructions for converting the DeepFashion Dataset to COCO format, simplifying object detection model training.

3. vis.py: Located in the detectron/utils/ folder, this file should replace the original vis.py in your repository. It adds functionality to crop test images based on predicted bounding boxes, as detailed in our paper.

4. Jupyter Notebooks: These notebooks contain the implementation details of Deep Convolutional Generative Adversarial Networks (DCGANs) as described in our paper. They include hyperparameters and network architectures for the generator and discriminator in PyTorch.

## Prerequisites
Before running the code, ensure you have the necessary dependencies and libraries installed. Key dependencies include Python, PyTorch, and required packages for data manipulation and visualization. You can install these dependencies using the command:

    pip install -r requirements.txt

## Training and Usage
1. **Data Preparation**:
* The dataroot variable specifies the root directory for your dataset. Make sure your fashion dataset is organized in this directory.
* It defines other parameters like the number of workers for data loading, batch size, image size, number of channels in the images (usually 3 for RGB), and other hyperparameters.

2. **Loading and Preprocessing Data**:
* [Dataset - Large-scale Fashion (DeepFashion) Database](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* It uses PyTorch's data loading functionality to load the dataset. Ensure that your dataset is in the specified directory structure.
* The code applies transformations such as resizing, center cropping, and normalization to the images.

3. **Generator and Discriminator Models**:

* The code defines the architecture for the GAN. It includes two key components: the generator and the discriminator.

* **Generator**:
        The generator is responsible for creating fake fashion images.
        It is implemented as a neural network with multiple layers.
        The Generator class is defined, including its layers for upsampling and batch normalization.

* **Discriminator**:
        The discriminator distinguishes between real and fake images.
        It is implemented as another neural network.
        The Discriminator class is defined, including its convolutional layers.

4. **Model Initialization**:

    It initializes the weights of the generator and discriminator using the weights_init function.
    Weight initialization can impact the training process significantly.
5. **Loss Functions and Training Parameters**:
    It sets up loss functions and training parameters.
    The **Binary Cross-Entropy Loss (BCELoss)** is commonly used for GANs.

6. **Training Loop**:
The training loop is the core of the GAN training process.
It iterates through the dataset and consists of two main steps:
   * **Discriminator Update**: In this step, the discriminator is trained. It learns to distinguish between real and fake images.
   * **Generator Update**: The generator is trained to create better fake images to fool the discriminator.
    The loss for both the discriminator and generator is calculated in each step.

7. **Monitoring Training Progress**:

    It saves generated images at regular intervals and keeps track of losses.
    This allows you to visualize the progress of your GAN during training.

8. **Saving Generated Images**:

    * After training, the code saves the generated images in the current directory for further analysis or use.

9. **Data Visualization**:
    * The code provides some functions for visualizing real and generated images.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

