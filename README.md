# RoboND-DeepLearning-Project
This project is using a Fully Convolutional Network (FCN) for semantic segmentation  in order to label the pixels of a person to follow  up (to identify and track a target) in images. A  deep neural network  has been trained to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

Files included: 
* [model_training.ipynb](/code/model_training.ipynb)
* A HTML version of [model_training.ipynb](model_training.html) notebook.
* Your model and weights file in the .h5 file format: [model_weights](data/weights)

## This project implemented the following steps:
### 1. Collecting the data 
There are data provided by Udacity, however for my model (Explained in the second step) that data was not enought to get the desirable accuracy. I added the following scenarios for the collected data:
* target is near in a dense crowd (images while following the target).
* no target only dense crowd (images while at patrol without target).
* the target is far in a dense crowd (images while at patrol with target).
![Alt text](/images/capture_data1.png)
![Alt text](/images/capture_data2.png)

### 2. Implement Fully convolutional Network for Semantic Segmentation (Understanding of the network architecture)
![Alt text](/images/fcn.png)
In a Fully Convolutional Layer (FCN) consist of  parts:

#### 1. Encoder
It is like a tipical Convolutional Neuronal Network (CNN) which has an input and its goal is to extract features, normally it ends in fully-connected layers for a classification, but in this case, it will end up with a 1x1 convolutional network, this is located before the decoder.

Each function includes batch normalization and it is using an activation function called a rectified linear unit (ReLU) applied to the layers. Check [Separable Convolutions and Encoder Block sectios](/code/model_training.ipynb) 

##### Why 1x1 convolutional was used instead of fully-connected layers
* It is more flexible because it allows different sized input images, instead of being fixed to one size
* It reduces the dimensionality of the layer, while preserving spatial information of the image, which allows us to output a segmented image
* It adds depth to our model and increases parameters at a fairly low computational price.

#### 2. Decoder
This one up-scales (upsample into) the output of the encoder for instance that this will be the same size as the original image. Check 
[Bilinear Upsampling and Decoder Block](/code/model_training.ipynb)  
The convolution layer  added is in order to extract more spatial information from prior layers. In the concatenation step, it is similar to skip connections. Using a skip connection is implemented in order to improve segmentation accuracy [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

At the end the activation function Softmax has been applied in order to generate the probability predictions for each of the pixels.

#### Bilinear Upsampling
Bilinear upsampling Bilinear interpolation (an extension of linear interpolation) The key idea is to perform linear interpolation first in one direction, and then again in the other direction, this in order to estimate a new pixel value. In this case, it uses the four nearest pixels.

#### Defined Model
This is the model generated, using the encoder and decoder, both united by a 1x1 Convolution layer:
```
plot_model (model, to_file='model.png', show_shapes=True,show_layer_names=True)
```
![Alt text](/code/model.png)

### 3.Train the model 
for the training it is using the following method:
```
model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)
```
#### chose hyper parameters
Those are the hyper parameters used:
```
learning_rate = 0.001 # value to be multiply with the derivative of the loss function
batch_size = 64 # the batch size is the number of training examples to include in a single iteration
num_epochs = 10# the number of epochs to train for 
steps_per_epoch = 400 # the number of batches per epoch
validation_steps = 100 # the number of batches to validate on 
workers = 2 #maximum number of processes.
```
I trained the FCN with 10 epoch in order to get the expected accurany. I ran the model in my using tensorflow and a GPU. The  batch size and learning rate are linked. If the batch size is too small then the gradients will become more unstable and would need to reduce the learning rate, in this case the batch size is 64. The hyper tuning performed is based on empirical validation. (I have tested with several combination of epochs, batch sizes, learning rates among others.)
This is the final result of epoch 10:

![Alt text](/images/epoch10.png)
```
400/400 [==============================] - 495s - loss: 0.0151 - val_loss: 0.0204
```
### 4.Check the score accuracy
* images while following the target
![Alt text](/images/following_target.png)
![Alt text](/images/following_target1.png)
* images while at patrol without target
![Alt text](/images/patrol_with_targer.png)
* images while at patrol with target
![Alt text](/images/patrol_without_target.png)

**The final grade score is  0.433361966648**

## Future Enhancements
* Adding more training data would be significant, this in order to avoid overfit and get more cases to learn.
* Test with a more architectures, generated by myself or using another proved network like [VGG, ResNet, GoogLeNet and so on](https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5) and then update them for Semantic Segmentation as it has been explained in [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).. 
