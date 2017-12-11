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
[image2]: ./myresult/yuv_y.png "Y image"
[image3]: ./myresult/normalized_image.png "normalized_image"
[image4]: ./myresult/1.jpg "Traffic Sign 1"
[image5]: ./myresult/2.jpg "Traffic Sign 2"
[image6]: ./myresult/3.jpg "Traffic Sign 3"
[image7]: ./myresult/4.jpg "Traffic Sign 4"
[image8]: ./myresult/5.jpg "Traffic Sign 5"
[image9]: ./myresult/my_visualize_cnn.png "visualize_conv1"

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

I used the numpy library to calculate summary statistics of the traffic
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

Here is an exploratory visualization of the data set. It is a bar chart showing how the data contribution by using pandas. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


1)change RGB to YUV color space for combining color image three channel edge information into the Y space(luma). So we can reduce the data number by using y space image and if use the U,V space, it can make image more unique.

![alt text][image2]

2)use a subtractive local normalization,and a divisive local normalization to extract image edge information( Reference: Pierre Sermanet and Yann LeCun.Traffic Sign Recognition with Multi-Scale Convolutional Networks.). The normolization is a process that changes the range of pixel intensity values to range 0 to 1. in some extent, its image result is similar to the brightness and contrast. and it is good for data sparsing and clustering.

![alt text][image3]

for additional datas,method used small translations, scaling  rotations, affine transformations, brightness, contrast and blur.
After test, i find straightly brightness and contrast may cause image deformation, small translations, scaling , rotations and affine transformations need take care of producing dark pixels. preprocess will be make further optimizing, in future(now result dont have good cluster result and dilute  data energe)



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     | 1x1 stride, same padding, outputs      |
|	                 |      32x32x108                                                               |
|                  	 |    Convolution1 3x3 	outputs 30x30x100                                        |
|                        |   Convolution2 3x3   outputs 30x30x8 	                                |
| RELU					| 			                                        |				
| Max pooling	      	| 2x2 stride,  outputs 15x15x108 			                	|
| Convolution 5x5	    | outputs 15x15x108 		|
| RELU					| 			                                        |				
| Max pooling	      	| 2x2 stride,  outputs 11x11x108                                                |
| Flatten				| outputs 2700		|						
| Fully connected		| outputs 400			|	
| RELU					|                       |
| Softmax				| outputs 200    	|		
| Fully connected		| outputs 200		|		
| RELU					|               |		
| Softmax				| outputs 100	|	


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an tensorflow and parameters:

| parameters        		|reason|
|:---------------------:|:---------------------:|
|EPOCHS = 10|using 10 can help tune model and show the train trend|
|BATCH_SIZE = 64| i also tried 128, 64 and 38 , but 128 and 38 cant enhance result like 64|
|mu = 0|it didnt change accurate if i changed numberr|
|sigma =0.05|i also tried 10, 1 , 0.05 and 0.0001 , but 10 , 1 and 0.0001 cant enhance result like 0.05|
|weight = tf.Variable(tf.truncated_normal())|i just use truncated_normal() which got values by cutting the random contribution|
|bias = tf.Variable(tf.zeros())|initializing bias by zero and using each epoch to update data |
|learning rate = 0.001|using 0.001 is not too small or large ,and can help to find better optimization|
|Optimiser = Adam(reference: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION) |adam is diagonal rescaling robust and fit for non-stationary problems )|


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

| Parameters|  
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

  1)image data need irrelative and dont have pixel miss as well as possible. Because image data make edge information beacome code for classify.
  
   2)convolution determine the filter type like sobel , wavelet etc. those make image energe focus on significant places.
   
   3)after try relu and tanh function, relu is more unstable than tanh continuous function. but  relu perfomance may great. So, in training process, unlinear is a good choice.
   
   4)ensure the each image or data is unique is important for training.
   
   5)Learning Rate is about to the optimization rate, if it is small, more time will be need.
   
   6)Network Topology. more layers will enhance training efforts. espacially, when you add extra info to the training data in layer like reference Traffic Sign Recognition with Multi-Scale Convolutional Networks sets two different operator in first layer.
   
   7)Batches and Epochs. batch depend on the computer calculate capable in some extent. epoch cant be too short, because training accurate is unstable and usually increase trend by number jitter.and there have high probability get local optimization. we need focus on accurate epoch change rules.
   
   8)Regularization. Regularization can helps to avoid overfit i.e. local optimization.
   
   9)Optimization and Loss. Optimization and Loss help function find optimization solution, like SGD , adam, etc. 
   
   10)Early Stopping. if find accurate is too small, review the image data whether right or not. if want accurate value become high, try change layers structure to make more image like human face diffrent condition image.
   
   11)Weight Initialization. weight value will change learn rate and make start accurate huge change. weight size is about details extraction. if size small , cnn will use fewer pixels to train which hold more small feature.

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
| 'Slippery road'			| 'Road narrows on the right'								|	
| 'Turn left ahead'	| 'Turn left ahead'						 				|
| 'Go straight or right'			| 'Go straight or right'						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares similar to the accuracy on the test set of small affine transform, noise and blur as 1.jpg, 2.jpg, 4.jpg. And results also shows model cannot deduce the whole image by some parts of image . when image have big image deformation like color change while shape not, the accurate is not some stable.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.8), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 12.067934			| 'General caution'				|						
| 9.3139172			| 'Wild animals crossing'					 |					
| 7.499558			| 'Road work'|
| 5.348145			| 'Traffic signals'					 |					
| 5.0472765			| 'Road narrows on the right'|
											

For the second image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 36.054581		| 'Ahead only'|									
| 9.9711342		| 'Turn right ahead'|							
| 7.1837049 	        | 'Go straight or left'|
| 3.1163545			| 'Keep left'					 |					
| 2.2545176			| 'Children crossing'|


For the third image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.4634168		| 'Road narrows on the right'		|						
| 3.4086015		| 'Road work'	|					
| 1.4387584	      | 'Slippery road'|
| 1.4270226			| 'Roundabout mandatory'					 |					
| 1.1857923			| 'Wild animals crossing'|


For the fourth image 
 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 16.780542		| 'Turn left ahead'|							
| 6.5700932		| 'Yield'|					
| 2.7808466	      | 'No entry'|
| 2.4858758			| 'Speed limit (30km/h)'					 |					
| 2.4331439			| 'Right-of-way at the next intersection'|


For the fifth image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 4.8400264		| 'Go straight or right'|								
| 4.2161355		| 'Road work'|							
| 2.6225283   	| 'Right-of-way at the next intersection'|
| 0.39990959			| 'Speed limit (80km/h)'					 |					
| 0.22273867			| 'Stop'|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

edge information, direction of image

![alt text][image9]


