# RoboND-DeepLearning-Project
This project is using a Fully Convolutional Network (FCN) for semantig segmentation  in order to label the pixels of a person to follow  up (to identify and track a target) in images. A  deep neural network  has been trained to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

## This project implemented the following steps:
### 1. Collecting the data 
There are data provided by Udacity, however for my model (Explained in the second step) that data was not enought to get the desirable accurancy. I added the following scenarios for the collected data:
* target is near in a dense crowd (images while following the target).
* no target only dense crowd (images while at patrol without target).
* the target is far in a dense crowd (images while at patrol with target).
![Alt text](/images/capture_data1.png)
![Alt text](/images/capture_data2.png)

### 2. Implement Fully convolutional Network for Semantig Segmentation
In a Fully Convolutional Layer (FCN) consist of  parts:

#### Encoder
It is like a tipical Convolutional Neuronal Network (CNN) which has an input and its goal is to extract features, normally it ends in fully-conected layers for a classification, but in this case it will end up with a 1x1 convolutional network wich is helping to reduce the dimensionality of the layer compared to a fully-connected layer layers.

Separable convolution layers consist of a convolution over each channel of an input layer, followed by a 1x1 convolution taking the output channels from the previous step and then combining them into an output layer. Separable convolution helps by reducing the number of parameters. The reduction in the parameters improves runtime performance. Separable convolution layers also reduce overfitting due to fewer parameters.

Each function includes batch normalization and it is using an activation function called a rectified linear unit (ReLU) applied to the layers. Check [Separable Convolutions and Encoder Block sectios](/code/model_training.ipynb) 

#### Decoder
This one up-scales  the output of the encoder for instance that this will be the same size as the original image. Check 
[Bilinear Upsampling and Decoder Block](/code/model_training.ipynb)  
The convolution layer  added is in order to extract more spatial information from prior layers. In the concatenation step, it is similar to skip connections. Using a skip connection is implemented in order to improve segmentation accurancy [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

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
learning_rate = 0.001 # 
batch_size = 64 #number of training samples/images that get propagated through the network in a single pass.
num_epochs = 10# the number of epochs to train for 
steps_per_epoch = 400 # the number of batches per epoch
validation_steps = 100 # the number of batches to validate on 
workers = 2 #maximum number of processes.
```
I trained the FCN with 10 epoch in order to get the expected accurancy. I ran the model in my using tensorflow and a GPU. The  batch size and learning rate are linked. If the batch size is too small then the gradients will become more unstable and would need to reduce the learning rate, in this case the batch size is 64. The hyper tuning performed is based on empirical validation. (I have tested with several combination of epochs, batch sizes, learning rates among otheres.)
This is the final result of epoch 10:

![Alt text](/images/epoch10.png)
```
400/400 [==============================] - 495s - loss: 0.0151 - val_loss: 0.0204
```
### 4.Check the score accurancy
* images while following the target
![Alt text](/images/following_target.png)
![Alt text](/images/following_target1.png)
* images while at patrol without target
![Alt text](/images/patrol_with_targer.png)
* images while at patrol with target
![Alt text](/images/patrol_without_target.png)

**The final grade score is  0.433361966648**

### 5.run  the model saved in the simulator
Check the following video:
