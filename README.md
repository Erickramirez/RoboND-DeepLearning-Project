# RoboND-DeepLearning-Project
This project is using a Fully Convolutional Network (FCN) for semantig segmentation  in order to label the pixels of a person to follow  up (to identify and track a target) in images. A  deep neural network  has been trained to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

![Alt text](/images/epoch10.png)
![Alt text](/images/following_target.png)
![Alt text](/images/following_target1.png)
![Alt text](/images/patrol_with_targer.png)
![Alt text](/images/patrol_without_target.png)


## This project implemented the folowing steps:
### 1. Collecting the data 
There are data provided by Udacity, however for my model (Explained in the second step) that data was not enought to get the desirable accurancy. I added the following scenarios for the collected data:
* target is near in a dense crowd (images while following the target).
* no target only dense crowd (images while at patrol without target).
* the target is far in a dense crowd (images while at patrol with target).
![Alt text](/images/caputure_data1.png)
![Alt text](/images/caputure_data2.png)

### 2. Implement Fully convolutional Network for Semantig Segmentation
In a Fully Convolutional Layer (FCN) consist of  parts:

#### Encoder
It is like a tipical Convolutional Neuronal Network (CNN) which has an input and its goal is to extract features, normally it ends in fully-conected layers for a classification, but in this case it will end up with a 1x1 convolutional network wich is helping to reduce the dimensionality of the layer compared to a fully-connected layer layers.

Each function includes batch normalization and it is using an activation function called a rectified linear unit (ReLU) applied to the layers. Check [Separable Convolutions and Encoder Block sectios](/code/model_training.ipynb) 

Why batch normalization?

Networks train faster because convergence is quicker, resulting in overall performance improvement.
Batch normalization allows higher learning rates, since the normalization helps gradient descent to converge more quickly.
Batch normalization adds some noise to the network, but works as well as dropout in improving performance.
Why separable convolution layers?

Separable convolution layers consist of a convolution over each channel of an input layer, followed by a 1x1 convolution taking the output channels from the previous step and then combining them into an output layer. Separable convolution helps by reducing the number of parameters. The reduction in the parameters improves runtime performance. Separable convolution layers also reduce overfitting due to fewer parameters.
--

#### Decoder
This one up-scales  the output of the encoder for instance that this will be the same size as the original image.

#### Defined Model
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
