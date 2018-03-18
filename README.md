# RoboND-DeepLearning-Project
This project is using a Fully Convolutional Network (FCN) for semantig segmentation  in order to label the pixels of a person to follow  up (to identify and track a target) in images. A  deep neural network  has been trained to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

![Alt text](/images/epoch10.png)
![Alt text](/images/following_target.png)
![Alt text](/images/following_target1.png)
![Alt text](/images/patrol_with_targer.png)
![Alt text](/images/patrol_without_target.png)
![Alt text](/images/preprocess_ims.py)


## This project implemented the folowing steps:
### 1. Collecting the data 
There are data provided by Udacity, however for my model (Explained in the second step) that data was not enought to get the desirable accurancy. I added the following scenarios for the collected data:
* the target is near in a dense crowd.
* the target is far in a dense crowd.
* the gartet has similar patern that the Quad.
**image**

### 2. Implement Fully convolutional Network for Semantig Segmentation
In a Fully Convolutional Layer (FCN) consist of  parts:
#### Encoder
It is like a tipical Convolutional Neuronal Network (CNN) which has an input and its goal is to extract features, normally it ends in fully-conected layers for a classification, but in this case it will end up with a 1x1 convolutional network wich is helping to reduce the dimensionality of the layer compared to a fully-connected layer layers.

#### Decoder
This one up-scales  the output of the encoder for instance that this will be the same size as the original image.
#### Define fcn_model
to conect encoder and deconder it is using a  Also, skip connection has been implemented in order to improve segmentation accurancy [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).
#### chose hyper parameters
![Alt text][1]
#### Save the model with its waights
### 3.Train the model 
### 4.Check the score accurancy
### 5.run  the model saved in the simulator



The following image shows the network architecture used;
```
plot_model (model, to_file='model.png')
```

I tested with several combination of epochs, batch sizes, learning rates among otheres.
