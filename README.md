## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
[//]: # (Image References)

[image1]: ./myresult/visualization.jpg "Visualization"
[image2]: ./myresult/yuv_y.png "Grayscaling"
[image3]: ./myresult/normalized_image.png "Random Noise"
[image4]: ./myresult/1.jpg "Traffic Sign 1"
[image5]: ./myresult/2.jpg "Traffic Sign 2"
[image6]: ./myresult/3.jpg "Traffic Sign 3"
[image7]: ./myresult/4.jpg "Traffic Sign 4"
[image8]: ./myresult/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 
  number:34799, size:32*32, channel:3

* The size of the validation set is ?
  number:4410, size:32*32, channel:3

* The size of test set is ?
  number:12630, size:32*32, channel:3

* The shape of a traffic sign image is ?
  32*32

* The number of unique classes/labels in the data set is ?
  43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data contribution.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


1)change RGB to YUV color space for combining color image three channel edge information into the Y space(luma)

![alt text][image2]

2)use a subtractive local normalization,and a divisive local normalization to extract image edge information( Reference: Pierre Sermanet and Yann LeCun.Traffic Sign Recognition with Multi-Scale Convolutional Networks.)

![alt text][image3]

for additional datas,method used small translations, scaling  rotations, affine transformations, brightness, contrast and blur.
After test, we find  brightness and contrast may cause image deformation, small translations, scaling , rotations and affine transformations need take care of producing dark pixels. preprocess will be make further optimizing, in future(now result is not cluster and dilute  data energe)



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     | 1x1 stride, same padding, outputs      
|	                 |      32x32x108                                                               |
|                  	 |    Convolution1 5x5 	outputs 28x28x100                                        |
|                        |   Convolution2 5x5   outputs 28x28x8 	                                |
| RELU					| 			                                        |				
| Max pooling	      	| 2x2 stride,  outputs 14x14x108 			                	|
| Convolution 5x5	    | outputs 28x28x108 		|
| RELU					| 			                                        |				
| Max pooling	      	| 2x2 stride,  outputs 14x14x108                                                |
| Flatten				| outputs 2700		|						
| Fully connected		| outputs 400			|	
| RELU					|                       |
| Softmax				| outputs 200    	|		
| Fully connected		| outputs 200		|		
| RELU					|               |		
| Softmax				| outputs 100	|	


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an tensorflow and parameters:
EPOCHS = 10
BATCH_SIZE = 64
mu = 0
sigma =0.05

weight = tf.Variable(tf.truncated_normal())
bias = tf.Variable(tf.zeros())

learning rate = 0.001


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
  99%
* validation set accuracy of ?
  97%
* test set accuracy of ?
  94%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
first architecture:
| Layer         		|  
|:---------------------:|
| EPOCHS = 10|
| BATCH_SIZE = 128 (too large GPU cannot calculate in respect with store, too small accurate is not percise)|
| Optimiser = Adam(reference: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION. adam is diagonal rescaling robust and fit for non-stationary problems )|
| mu = 0|
| sigma =0.1(depend on the normalized image pixel value)|
| layer1 convolution just have one image channel and simple normalized oparetor|
 

* What were some problems with the initial architecture?
1)avalidation accurate is low
2)preprocess image edge is broken
3)sigma will effect weight change

* How was the architecture adjusted and why was it adjusted? 
1)change preprocess method and make result image dont deform 
2)make sigma around 0.5

Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
  sigma: change weight, make convolution different like sobel
  batch_size: make large database smaller 
  convolution layer size: make edge information different 

  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  1)image data need irrelative and dont have pixel miss as well as possible. Because image data make edge information beacome code for classify
   2)convolution determine the filter type like sobel , wavelet etc. those make image energe focus on significant places.
   3)after try relu and tanh function, relu is more unstable than tanh continuous function. but  relu perfomance is great. So, in training process, unlinear is a good choice.
   4)ensure the each image or data is unique is important for training

If a well known architecture was chosen:
* What architecture was chosen?
  Lenet
* Why did you believe it would be relevant to the traffic sign application?
  sign like digit number dont have deformations

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  ensure enough edge , luma and shape information in training data 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

|imageID|image name|
|:---------------------:|:---------------------------------------------:| 
|1.jpg| 'General caution'|
|2.jpg| 'Ahead only'|
|3.jpg| 'Slippery road'|
|4.jpg| 'Turn left ahead'|
|5.jpg| 'Go straight or right'|

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because  image exists small affine transform. 
The second image might be difficult to classify because  image exists watermark noise . 
The third image might be difficult to classify because  the most parts of image is snow and we need to deduce the image meaning from rest of parts. 
The fourth image might be difficult to classify because  image exists shadow blur.
The fifth image might be difficult to classify because  image exists inhomogeneous color change.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 'General caution'		| 'General caution'									| 
| 'Ahead only'			| 'Ahead only'									|
| 'Slippery road'			| 'Bumpy road'											
| 'Turn left ahead'	| 'Turn left ahead'						 				|
| 'Go straight or right'			| 'Keep left'						|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares similar to the accuracy on the test set of small affine transform, noise and blur as 1.jpg, 2.jpg, 4.jpg. And results also shows model cannot deduce the whole image by some parts of image and cannot fight  with big image deformation like color change while shape not.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 18.267069			| 'General caution'				|						
| 11.561676			| 'Double curve'					 |					
| 7.834867			| 'Road narrows on the right'|
| 7.439065			| 'Children crossing'					 |					
| 6.7417784			| 'Speed limit (60km/h)'|
											

For the second image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 28.280853		| 'Ahead only'|									
| 6.9413133		| 'Road narrows on the right'|							
| 5.0935559 	        | 'Yield'|
| 4.9019399			| 'Turn right ahead'					 |					
| 4.8930273			| 'Speed limit (60km/h)'|





For the third image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 2.4563167		| 'Bicycles crossing'		|						
| 2.2535417		| 'Road narrows on the right'	|					
| 1.8545729	      | 'Beware of ice/snow'|
| 1.4050156			| 'Wild animals crossing'					 |					
| 1.2658368			| 'Road work'|

For the fourth image 
 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14.052359		| 'Turn left ahead'|							
| 7.2908397		| 'Speed limit (30km/h)'|					
| 5.6769919	      | 'Road narrows on the right'|
| 5.2392116			| 'Dangerous curve to the left'					 |					
| 4.8996344			| 'Go straight or left'|


For the fifth image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 4.3794479		| 'Children crossing'|								
| 3.0187306		| 'Speed limit (60km/h)'|							
| 1.8864288   	| 'Bicycles crossing'|
| 1.7862785			| 'Speed limit (80km/h)'					 |					
| 1.5917232			| 'Dangerous curve to the right'|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

edge information, direction of image



