#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/samples.png "Samples"
[image2]: ./writeup_images/histogram.png "Histogram"
[image3]: ./writeup_images/widths.png "Widths"
[image4]: ./writeup_images/heights.png "Heights"
[image5]: ./writeup_images/newimages.png "New Images"
[image6]: ./writeup_images/softmaxbar.png "Softmax Bar"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pengjoshua/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set: 34799 example images
* The size of test set: 12630 example images
* The size of validation set: 4410 example images
* The shape of a traffic sign image: (32, 32, 3), meaning 32 pixels wide, 32 pixels high, 3 RGB color channels 
* The pickled data (which we load) contains resized versions (32 by 32) of the original images.
* The number of unique classes/labels in the data set: 43 classes/labels

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here are some sample training set traffic sign images with their labels.

![alt text][image1]

Here is an exploratory visualization of the data set. It is a histogram of the various traffic sign classes in the training set. The mean label count across the training set is 809.3 and the median label count 540.0.

![alt text][image2]

The following plot is a histogram of image widths of the original traffic sign images (before resizing). The min traffic sign width is 25 px and the max is 243 px.

![alt text][image3]

The following plot is a histogram of image heights of the original traffic sign images (before resizing). The max traffic sign height is 25 px and the max is 225 px.

![alt text][image4]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. 

Essentially, scaling the inputs through normalization gives the error surface a more spherical shape, where it would otherwise have a very high curvature ellipse. Since gradient descent is curvature-ignorant, having an error surface with high curvature will mean that we take many steps which are not necessarily in the optimal direction. When we scale the inputs, we reduce the curvature, which makes methods that ignore curvature (such as gradient descent) work much better. When the error surface is circular (spherical), the gradient points right at the minimum, so the algorithm converges to an optimal solution (model) more quickly. Here I apply min-max normalization which scales the data between -0.5 and 0.5.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. 

I decided to augment the training set with randomly rotated (+/- 20 deg) and randomly translated copies (+/- 5 px). I added the rotated and translated images to the training set to make the upcoming model trained on this data more robust to similar potential variations in the testing set.

The difference between the original data set and the augmented data set is:
* addition of randomly rotated images (+/- 20 deg)
* addition of randomly translated images (+/- 5 px)
* min-max normalization (-0.5, 0.5)

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6  					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16  					|
| Fully connected		| outputs 1024x1       							|
| RELU					|												|
| dropout				| keep-prob 0.5 								|
| Fully connected		| outputs 1024x1       							|
| RELU					|												|
| dropout				| keep-prob 0.5 								|
| Fully connected		| outputs 1024x1       							|
| RELU					|												|
| dropout				| keep-prob 0.5 								|
| Fully connected		| outputs 43x1        							|
| L2 regularization		| penalty 0.001        							|
| Softmax				|        										|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a modified version of the LeNet convolutional neural network with mean = 0 and standard deviation = 0.01.

I start with a convolutional layer with a 5x5 filter with an input depth of 3 and an output depth of 6 and initialize the bias. Using the following formulas, I compute the output height and output width of the filter. The output vector is 28x28x6.

new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1

Then I use the conv2d function to convolve the filter over the images and add the bias at the end. Next, I activate the output of the convolutional layer with a ReLU activation function. 

A Rectified linear unit (ReLU) is type of activation function that is defined as f(x) = max(0, x). The function returns 0 if x is negative, otherwise it returns x. The ReLU activation function effectively turns off any negative weights and acts like an on/off switch. Adding additional layers after an activation function turns the model into a nonlinear function. This nonlinearity allows the network to solve more complex problems.

Then, I apply max pooling with a 2x2 stride which produces a pooling vector output of 14x14x6.

Conceptually, the benefit of the max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining values.

I add a second convolutional layer with a 5x5 filter producing an output vector of 10x10x16. Once again, I use conv2d, add the bias, apply ReLU activation, and apply max pooling with 2x2 stride to generate an output vector of 5x5x16.

Next, I flatten the vector and pass it into a fully connected layer with a width of 1024. Then, I apply a ReLU activation function to the output of this fully connected layer. Next, I apply dropout with a keep-prob of 0.5. 

Dropout is a regularization technique for reducing overfitting. The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections. keep-prob allows you to adjust the number of units to drop. In order to compensate for dropped units, tf.nn.dropout() multiplies all units that are kept (i.e. not dropped) by 1/keep-prob.

I repeat this pattern for 3 more fully connected layers and output a fully connected layer with a width equal to the number of classes = 43. The output of this entire LeNet/convnet function is also referred to as logits.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Using 100 training epochs, keep-prob of 0.5, learning rate of 0.001, batch size of 128...

My final model results were:
* training set accuracy of 96.5%
* validation set accuracy of 96.5% 
* test set accuracy of 94.8%
* new images accuracy of 100.0%

The AdamOptimizer was used for training with an initial learning rate of 0.001. I utilized learning rates 0.01, 0.005, 0.002, 0.001, and 0.0005 and found that higher learning rates would reach high validation accuracies earlier but then oscillate/overshoot and fall back down in later epochs. Lower training rates would progress very slowly and did not achieve very high accuracies at the end of 100 epochs. The initial learning rate of 0.001 turned out the be the optimal learning rate for my custom convnet. 

Adding a fourth fully connected layer, increasing the batch size to 128, and resizing the width of all fully connected layers to 1028 increased the training and validation accuracies.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 25 German traffic signs that I found on the web:

![alt text][image5] 

These are german traffic sign images I pulled from this website: http://www.gettingaroundgermany.info/zeichen.shtml

All of these images are clean, ideal images of german traffic signs. Although it is a bit counterintuitive and a reverse way of thinking, I want to see how a convnet trained on real life images would handle clean, perfect images. Usually, it is the other way around, you train a model on clean, neat, organized data because that's all you have and the resulting model you've trained tends to make misclassification errors on real world, disorganized, messy data. I want to investigate the reverse situation where you train on messy data and evaluate your model on clean data. I expect that my convnet would perform very well with clean images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image											| Prediction									| Probability		|
|:---------------------------------------------:|:---------------------------------------------:|:-----------------:| 
| Speed limit (60km/h)							| Speed limit (60km/h)							| 0.989564			|
| No passing									| No passing									| 0.999982			|
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons	| 1.0				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection			| 0.999496			|
| Yield											| Yield											| 1.0				|
| Stop											| Stop											| 0.997529			|
| Vehicles over 3.5 metric tons prohibited		| Vehicles over 3.5 metric tons prohibited		| 1.0				|
| No entry										| No entry										| 1.0				|
| General caution								| General caution								| 0.99948			|
| Dangerous curve to the left					| Dangerous curve to the left					| 0.965671			|
| Dangerous curve to the right					| Dangerous curve to the right					| 0.654329			|
| Double curve									| Double curve									| 0.999983			|
| Bumpy road									| Bumpy road									| 0.999871			|
| Slippery road									| Slippery road									| 0.999626			|
| Pedestrians									| Pedestrians									| 0.99804			|
| Children crossing								| Children crossing								| 0.999996			|
| Bicycles crossing								| Bicycles crossing								| 0.841453			|
| Beware of ice/snow							| Beware of ice/snow							| 0.997745			|
| Wild animals crossing							| Wild animals crossing							| 0.998305			|
| Keep right									| Keep right									| 1.0				|
| Keep left										| Keep left										| 0.999959			|
| Roundabout mandatory							| Roundabout mandatory							| 0.999824			|
| End of no passing								| End of no passing								| 0.997461			|

The model was able to correctly guess all 25 of the 25 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.5%. Given that the new images I found were all clean, ideal images, I expected an extremely high prediction accuracy. I expect that my convnet model will have difficulty with data including non-german traffic signs, grayscaled images removing color channel information, and additional text/graffiti on traffic signs.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following are bar charts of the top 5 softmax probabilities for each of the 25 new images. Most signs had one dominant softmax probability close to 100% accurate. Only "Double curve" and "End of no passing by vehicles over 3.5 metric tons" had any significant competition from other classes. There are 2 variations of the double curve sign with the zig-zag facing the opposite direction. The "End of no passing by vehicles over 3.5 metric tons" sign is grayscaled already and is quite similar in appearance to "End of no passing zone" and "End of all Restriction signs".

![alt text][image6]
