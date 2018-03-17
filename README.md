# RoboND-DeepLearning-Project
This project is using a Fully Convolutional Network (FCN) for semantig segmentation  in order to label the pixels of a person to follow  up (to identify and track a target) in images. A  deep neural network  has been trained to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

In a Fully Convolutional Layer (FCN) consist of  parts:
* Encoder: it is like a tipical Convolutional Neuronal Network (CNN) which is useful for classification. The goal is to extract features.
* Deconder: this one up-scales  the output of the encoder for instance that this will be the same size as the original image.
Also, to conect encoder and deconder it is using a 1x1 convolutional network wich is helpeping to reduce the dimensionality of the layer compared to a fully-connected layer layers. Also, skip connection has been implemented in order to improve segmentation accurancy [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

## This project implemented the folowing steps:
### 1. Collecting the data 
### 2. Implement Fully convolutional Network for Semantig Segmentation
#### Define the encoder block
#### Define decoder block
#### Define fcn_model
#### chose hyper parameters
#### Save the model with its waights
### Train the model 
### Check the score accurancy
### run  the model saved in the simulator



The following image shows the network architecture used;
```
plot_model (model, to_file='model.png')
```

I tested with several combination of epochs, batch sizes, learning rates among otheres.
