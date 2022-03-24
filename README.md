# Deep Learning Mini Project 1:
## Submission by:
* Name: Disha Lamba <br/>
Netid: dl4747 <br/>
* Name: Evan Lehrer <br/>
netid: el3294 <br/>
* Name: Junda Wu<br/>
  netid: jw6466 <br/>
  
 # Mini Project 1 Topic:
 ## **Objective:** <br/>
 Design a ResNet architecture to maximize accuracy on the CIFAR-10 dataset under constraints. <br/>
(1) maximize test accuracy on CIFAR-10 while ensuring our model has <br/>
(2) less than 5M trainable parameters.

## About CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. The dateset has 50000 training images and 10000 test images.
</br>
Here are the classes in the dataset with some random images from each class:
</br>
<img src="https://imgs.search.brave.com/mrVWRVhArBOD1FmbyACPHGplehLu_QeasvSPLeIFQc0/rs:fit:527:424:1/g:ce/aHR0cHM6Ly9zaWNo/a2FyLXZhbGVudHlu/LmdpdGh1Yi5pby9j/aWZhcjEwL2ltYWdl/cy9DSUZBUi0xMF9l/eGFtcGxlcy5wbmc" alt="MarineGEO circle logo" style="height: 500px; width:500px;"/>

## Implementation
To achieve the objective, our implementation and study were focused on rescaling the vanilla ResNet-18 using PyTorch. 
### Architecture:
The predefined hyperparameter sets, which are changeable in our problem setting, are related to the design of the
ResNet model including model depth, model width and kernel sizes. The hyperparameter sets are as
following: </br>
N : Residual Layers </br>
Bi: Residual blocks in Residual Layer i </br>
Ci: Channels in Residual Layer i </br>
Fi: Conv. kernel size in Residual Layer i </br>
Ki: Skip connection kernel size in Residual Layer i </br>
P: Average pool kernel size </br>
<br>
![Architecture](https://user-images.githubusercontent.com/26017359/160009325-1b29a0cb-3702-41c8-96a8-a6961eb2609e.png)
</br>
In our model architecture design, C_1 is the essential hyperparameter to control the width of the ResNet model. N and B<sub>i</sub> are jointly deciding the model depth, but in a different way. By generally increasing B<sub>i</sub>, each residual layer is deeper, which makes the skip connection relatively less frequent. N is limited to less than 6 in our design, since the input resolution and the stride step are fixed. In addition to model width and depth, kernel sizes in the residual layer F<sub>i</sub> and the kernel sizes of the skip connection layer K<sub>i</sub> determinate the perceptive field of the convolutional networks. Finally, the average pooling size P controls the complexity of the final feed-foward layer. Basically, the purpose of controlling P is to find out how much details from the final feature map should be preserved.

In our approach, we have used
  * **Regularization** - To prevent potential overfitting problems for both convolutional networks and the last fully connected layer, we introduce two dropout mechanisms, **DropBlock and feature dropout**. The feature dropout layer is applied after the final average pooling layer.
  * **Data Augmentation** - To further enable the model to be generalized to the test dataset, we introduce crop, resize and flip in the training stage. Since the CIFAR-10 image resolution is in a relatively small size of 32x32, the cropped local patch can be unrecognizable due to too low resolution. Thus, we implement this augmentation mechanism by cropping a same 32x32 size of image patch by zero padding of $4$ pixels on both sides.
  * **Some Training Techniques** - We fix the **batch size of training samples to 64** and reshuffle the whole training set in each epoch. We use **Adam optimizer and set the initial learning rate to 0.001**. The **total training epochs are set to 60**.
  
Note: More details about each technique can be found in our report. 

## Results:
### Hyperparameter setting:
After Compound scaling which is inspired from [EfficentNet 2019](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf) paper, we get our hyperparameter values as:</br>
N = 2, </br>
C1 = 128, </br>
P = [1,1], </br>
B = [3,3], </br>
F = 3, </br>
K = [1,1] or [3,3] or [5,5]. </br>

### Accuarcy and Number of parameters:
After rescaling our model and setting the hyperparameters we get our accuracy as 90.70% with M parameters.


## Repository Use Instructions:


