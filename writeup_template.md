**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

You're reading it! and here is a link to my [project code](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

# Dataset Exploration #

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630.
* The shape of a traffic sign image is 32X32
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for each of the 43 classes on train and test datasets.

![Class Distribution](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/distr_class.png)

From the previous plot we can tell that test and traininig dataset have same distribution which allows our model generalize better.
On the other, both train and test dataset have quite unbalanced classes, i.e., the class distribution is not uniform which means the model will more likely to have higher accuracy to certain classes.  Maybe during our model we should try to balanced classes to obtain higher model accuracy.


# Design and Test a Model Architecture #
 ## Preprocessing ##

As a first step, I decided to convert the images to grayscale because it was important the level of image complexity from 3 color channels to 1. This will help the model to focus more on shape and lines of images instead trying to figure out the difference between
different color layers.As result, we will get a 2D matrix  representing the image. The value of each pixel in the matrix will range from 0 to 255 – zero indicating black and 255 indicating white.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale Transformation](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/GrayscaleTransform.png)

As a second step, I decided to use histogram equalization technique because it allows grayscale images to show more distinct images. Since histogram equalization spreads out intensity values along the total range of values in order to
achieve higher contrast, this will help our model to classify better since it evidences traffic signs edges more clearly.

Here is an example of a traffic sign image before and after histogram equalization on the grayscale images.

![Histogram_Equalization Transformation](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/HistEquTransform.png)

As a last step, I normalized the image data because Deep learn networks learn faster for normalized input features . In addition, the pixel intensity will have more similar range values and as result it will help to find more suitables weights. It is also important because since we will be using weight sharing whereas similar range input values will have an extremelly important role.

## Augment the training data ##

I decided to generate additional data because my model was overfitting. 
Here it is a plot of loss and accuracy before augmenting data.

![Loss_Accuracy_Initial](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/before_augmented_Data.png)

The accuracy for training data is larger than test one. Clearly, this is a strong indication of overfitting. In addition, the loss for test data is larger than training one, which confirms our assumption of overfitting. 
After augmenting the training data, I obtained the following loss and accuracy plots:

![Loss_Accuracy_Augmented](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/LossTrainVsLossValidationAugDeep2.png)

From previous plot, I can tell that the model is not overfitting anymore since the accuracy for test data increase and overtook training data one. Although, the error for training data set increased, which related with distortation applied during augmented data generation. In further detail, since the images are twisted it will increase the error, but on the other hand, it will train better the model to recognize the same image under different image properties.

To add more data to the the data set, I used the following affine transformations:
* Random rotation between -22.5 and 22.5 degrees
* Random shearing between -5 and 5 shift on (x,y) pixel position
* Image translation between -1 and 1
* Random brightness increase between 0 and 1

Here is an example of an original image and an augmented image:

![Augmented_Data](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/AugmenteData.png)

The difference between the original data set and the augmented data set is that the number of samples just doubled to 69598.The distribution after generating the augmented dataset was the following:

![Augmented_Data_class_dist](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/distr_class_aug.png)

The distribution of classes hold the same profile as expected. In further detail, I generated a new augmented dataset keeping in mind not to change initial dataset distribution and apply a different transformation for each image. Although, as a future work, generate augmented data in order to balance each class will be a clear option.

## Model Architecture ##

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10X10X16|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 5x5x64 |
| Fully connected	1	| 576 input weights,     120 hidden neurons			|
| DROPOUT					|						Probability = 0.7							|
| Fully connected	2	| 120 input weights,     84 hidden neurons	       									|
| DROPOUT					|					Probability = 0.7								|
| Fully connected	3	|     84 input weights,    43 output classes    									|
| DROPOUT					|					Probability = 0.7							|
| Softmax				|      43 classes probability   									|
|						|												|
|						|												|

## Model Training #

To train the model, I used an Adam optimizer with a batch size of 128 samples and 10 epochs. 
In order to measure model performance, I used cross entropy to measure model error and accuracy. Both complement each other, giving a more broad view how the model generalizes.
Regarding the hyperparameters initialization, the learning rate used was 0.001 and a gaussian distribution with mean zero and 0.1 standard deviation was used to initialized DNN weights.

## Solution approach ##

My final model results were:
* training set accuracy of 90 %
* validation set accuracy of 96,5 % 
* test set accuracy of 92,6 %

The first used architecture was LeNet one provided by Yann LeCun paper, due to the good performance in 32X32 pixels classification. 
This initial architecture was initially overfitting. Therefore, I decided to apply dropout regularization using probability of 0.7 , i.e., 30% of neurons dropped. There was an improvement in performance when using a bit of dropout on convolutional layers, thus left it in, but kept it at minimum.
Although, the achieved accuracy for validation set was below expectations , i.e., it was approximately 93%. As a result, I readjust the architecture by adding one convolution layer and as a result increase the model depth to 32 layers. This change lead to an increase of accuracy around 3,5 % in validation set. My reasoning behind this convolution was trying to increase model perception of traffic signs shape detail.
Therefore, important design option in a model are the number of convolution layers and their depth in order to highlight important features and reduce the number of parameters learning. In addition, the number of fully connected allows the network to learn, so the number of layers influence how well a model performs.

# Test a Model on New Images #
## Acquiring new images ##
From a quality image standard, all the images have a reasonable one. Although, for Yield sign I antecipate difficulties to the model because of the sign surroundings and its almost 3D perspective.

Here are five German traffic signs that I found on the web:

![sign_1](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/Images/Sign1.png) ![sign_2](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/Images/Sign2.jpg) ![sign_3](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/Images/Sign3.jpg)
![sign_4](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/Images/Sign4.jpg) ![sign_5](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/Images/Sign5.jpg)

The first image might be difficult to classify because the model isn't trained well with respect to 60kmph class. It also depends on the particular image if a particular image is off in the standards of trained model then it won't predict properly and that's why we train the model on various versions of the same data point.

## Performance on New Images ##

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)    		| Speed limit (80km/h)  									| 
| Stop   			| Stop 										|
| Yield					| Yield											|
| Speed limit (50km/h)    		| Speed limit (50km/h)			 				|
| No entry			| No entry     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of comparing the number of right classification images. Although, since the test dataset accuracy is over 90% and the model get 80 % accuracy for web images, this means the constraint on the model get an accuracy 100% for small number of images  as well as a poor training in this particulary class. 

## Model Certainty - Softmax Probabilities ##

The code for making predictions on my final model is located in the 22th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit (60km/h)  sign (probability of 0.8), and the image does doesn't contain the right speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80         			| Speed limit (80km/h)  		 									| 
| .34     				| Stop 	 										|
| .93					| Yield												|
| .98	      			| Speed limit (50km/h)				 				|
| .99				    | No entry    							|


For the second image, the model has a low probability (0.34) for stop sign, but it predicted it right. 
Regarding the rest of images, there is a high probability(>0.9) and the prediction is the right one.
Finally, it follows the top 5 softmax probabilities per sign:
![sign_1](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/Hist0.png) ![sign_2](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/Hist1.png) ![sign_3](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/Hist2.png)
![sign_4](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/Hist3.png) ![sign_5](https://github.com/BrunoEduardoCSantos/SelfDrivingCarNanodegreeUdacity/blob/master/P2-Traffic-Sign-Classifier/WriteUpImages/Hist4.png)

