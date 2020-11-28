# Digit Recognition On Raspberry Pi
## REAL-TIME HANDWRITING RECOGNITION USING TENSORFLOW AND RASPBERRY PI

<p align="center">
<img src="/images/before.PNG">
</p>

<p align="center">
<img src="/images/after.PNG" height=235> 
</p>

<p align="center">
<img src="/images/summary.PNG">
</p>

## Description
This project can recognize handwritten digits in real-time. The convolutional neural network (CNN) was trained on the Raspberry Pi using the MNIST dataset. During operation, a handwritten digit is placed in front of the camera, and the predicted digit is displayed on the Sense HAT LED panel along with the model’s confidence level. The captured images must be resized to 28x28, converted to grayscale, and inverted to match the format of the training images.

## Hardware
I used a Raspberry Pi 4B with 2GB of RAM, the Sense HAT, and Pi Camera Module V2.

## Packages
These are the python packages that needed to be installed on the Raspberry Pi:
- Numpy
- OpenCV
- TensorFlow
- PiCamera
- SenseHat

## Demo Video
Coming soon - I will post the demo video on YouTube and provide a link.

## History
This was a semester final project for my Neural Networks & Machine Learning elective course. Throughout the class, I learned how to use many python packages including Keras, Pandas, Numpy, and OpenCV. I learned the math behind linear regression, logistic regression, Relu and Sigmoid activation, and CNNs. The purpose of this project was to utilize most of the skills and knowledge from the class and implement it on an embedded system for a practical application.
