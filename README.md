# Build a Traffic Sign Recognition Program #

## Overview  ##
In this project, a deep neural networks and convolutional neural networks was implemented to classify German traffic signs. The model was trained and validated using  [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, the model was tested to to identify images of German traffic signs around the web.

The base code is contained in the following IPython notebook: [Traffic_Sign_Classifier](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

A detailed description of the project is on writeup file: [writeup_template](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/writeup_template.md). 


## Setup ##

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

# Usage

Run the following command to clone project:

```sh
git clone https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier
jupyter notebook Traffic_Sign_Classifier.ipynb
```

## Softmax probabilities for 5 German Traffic Signs ##
![sign_1](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/WriteUpImages/Hist0.png) ![sign_2](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/WriteUpImages/Hist1.png) ![sign_3](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/WriteUpImages/Hist2.png)
![sign_4](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/WriteUpImages/Hist3.png) ![sign_5](https://github.com/BrunoEduardoCSantos/Traffic-Sign-Classifier/blob/master/WriteUpImages/Hist4.png)


# Disclamer
This project was cloned from [Udacity Traffic Sign project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project) in the context of [Self-Driving Carnanodegree](https://eu.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).
