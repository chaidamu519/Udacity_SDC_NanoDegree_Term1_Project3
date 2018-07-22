# Behavioral Cloning Project

This repository contains starting files for the Behavioral Cloning Project.




####  Goals

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




####  Project Files

My project includes the following files:

* [drive.py](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/drive.py) for driving the car in autonomous mode ( added one line for the conversion from RGB to BGR )

* [model.h5](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/model.h5) containing a trained convolution neural network 

* [model.py](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/model.py) The script used to create and train the model.


* The documents in the folder [Model 1](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/tree/master/Model%201) are the results obtained without data augmentation.

* [run3.mp4](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/run3.mp4) recorded video.



### Model Architecture and Training Strategy

#### 1. Data Collection

Learning strategy was chosen to keep the vehicle driving on the road and to balance the dataset between straight lines, curves, and recovery from side of the road.

* Clockwise: center lane driving (2 laps), recovering from the left and right sides of the road (2 laps), focusing on driving  around curves (2 laps).

* Counter Clockwise: center lane driving (2 laps), recovering from the left and right sides of the road (2 laps), focusing on driving  around curves (2 laps).

The data distribution is shown as the following:
![alt text](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/Model%201/distribution.png)


### 2. Data Preprocessing and Augmentation

* All the images obtained from left, right and center cameras are directly used. The final dataframe combines two clumns: all the path names of the images from three cameras and the corresponding steering angles. A correction factor is added on the steering angles of the left and right cameras. The correction factor is tuned as a hyperparameter during training.

* Image cropping and normalization are performed in Keras model

* Brightness variation.
    The brightness of the image is varied between 0.5 and 1.5 times of the original HLS to simulate different conditions

* Image Horizontal Flipping. The steering angles are flipped as well.
   ###### Center Camera
![alt text](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/Image/Center_Flip_2.png)

   ###### Left Camera
![alt text](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/Image/Left_Flip_1.png)

   ###### Right Camera
![alt text](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/Image/Right_Flip_2.png)


### 3. Generator

Since the dataset is extremely large. I use a python generator to perform real-time data augmentation instead of storing the preprocessed data in memory all at once.

#### 4. Model Architecture

* I followed directly the CNN architecture used by [NVIDIA](https://arxiv.org/pdf/1604.07316.pdf) in their paper and after training the car can stay on the track without data augmentation. 
![alt text](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/Model%201/Screen%20Shot%202018-07-22%20at%205.11.23%20PM.png)

* Three Dropout layers were added after the Flatten layer and two fully-connected-layer to reduce the overfitting. These factors are tuned during training as hyperparameters.

* The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle can stay on the track.

* The model used an adam optimizer, so the learning rate was not tuned manually.

* Mean Squared Error loss was used as the loss metrics for traning and validation dataset.

#### 5. Test in simulator

Since the cropping and normalization are performed in the keras model. Only one line of code wass added in [drive.py](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project3/blob/master/drive.py)

