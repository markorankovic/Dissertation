# Dissertation
 
All Python work found at the link: https://drive.google.com/open?id=1uNsPD9GqtPR8wtZj4jx-hkBalJg5YiHU

# Using Machine Learning to Gather High Resolution Face Images for Face Super Resolution

Marko Rankovic
Computer Science (MEng) at Leeds Beckett University

## Table of Contents

## Figures

## Acknowledgements
My father for his knowledge in super resolution, supervision of Rahman, George who always checked on me and the deterministic universe should it accidentally fall on my side for any positive outcome of this thesis.

## Abbreviations

ML - Machine Learning

SR - Super Resolution
LR - Low Resolution
HR - High Resolution

GANs - Generative-Adversarial Networks
SRGANs - Super Resolution Generative-Adversarial Networks
WGANs - Wasserstein Generative-Adversarial Networks
FSRGANs - Face Super Resolution Generative-Adversarial Networks

CNN - Convolutional Neural Network

## Abstract
This paper delves into the most optimal method of taking a face image in low resolution e.g. a webcam image, and reconstructing the image in high resolution using a specific form of generative-adversarial networks.

## Introduction
In this thesis, an application will be made which can take a low resolution (LR) face image as input, process the image through a neural network and then output a higher resolution (HR) version of the image. The exact machine learning (ML) algorithms to use, the loss functions, optimizers and activation functions alongside the neural network model will be determined through a practical, theoretical and critical analysis through the existing techniques. An investigation will be undertaken into how different algorithms of ML may be more or less practical from scenario to scenario, but most importantly for constructing super resolution (SR) face images.

In this project applications of super resolution does not fit for scientific purposes which require great levels of accuracy such as forensic science, so for instance a suspect in a CCTV footage would not be good to upscale using machine learning as there are chances that a misrepresentative face may result. Outcomes of face super resolution in this project are solely for emotional value.

## Literature Review
In a general sense of reconstructing images in higher resolution, generating a larger size version of an image is not to be confused with upscaling, which often involves multiplying the number of each pixel of the original image. This only shows the same image looking as though it consists of larger pixels rather than a more detailed image, as discussed by [1]. He discusses how the most honest method of achieving a larger image without retaking the image is using the concept of super resolution. Super resolution often involves the usage of machine learning to feed the model a smorgasbord of familiar images so the computer has reference for what individual objects in the image to be upscaled look like in close-up. Training images for the model are important. They need to be as relevant as possible, which is why for super-resolution a good dataset would be high resolution images of similar context downscaled. The Convolutional Neural Network (CNN) takes as the input low resolution images and learns to reproduce the original high resolution images.

A CNN is very efficient method of training images after distinct features. A convolution in this scenario works by going to each pixel whose new a value will be a sum of all the neighboring values multiplied by the corresponding filter value in a 3x3 filter matrix [5]. As a result, many trivial features of the image is filtered out to leave behind those worthy of processing.

The mean squared error (MSE) optimizer for super resolution does not work well due its focus on the differences of the training image and its original on a pixel level which excludes the overall structure of the image to be upscaled [3]. This is where the structural similarity index replaces MSE, as this optimization takes into account of the structural content of the image. 

For deep learning as a method for super resolution, [2] explains how the result by using the deep learning model tends to be less realistic and more exaggerated, looking more like a photoshop edit unlike the use of generative-adversarial networks (GANs) which consists of two separate neural networks training simultaneously, which one responsible for generating the image to challenge the other in telling apart from real and fake images. GANs for super resolution applications are called SRGANs.

For face super resolution, Wasserstein GANs (WGANs) have been made usage of by [4] as the Wasserstein metric compared with other objective functions has shown to be the most effective for training GANs to perform face SR.

VDSR, URDGN and SRResNet all result with relatively blurry faces in comparison to FSRNet and FSRGAN as demonstrated in [6]. FSRNet stands out from the other SR models by first examining the geometry of the face using heatmaps for facial landmarks, which makes it possible to super resolve low-resolution images without the need for facial alignment.

With regards to [7], a 2D image for a computer for machine learning applications will be read as a three dimensional matrix since each pixel of the higher 2D array will be represented as an array of red, blue and green values ranging from 0 to 255. This means that a 800 x 600 image would be read as an array 800 x 600 x 3. What the computer however makes of these pixel values could drastically vary between differing machine learning algorithms.

According to [8] about the image and its dimensions, an image consists of three separate matrices each corresponding to red, green and blue. This contradicts the idea of a 3D array with a 2D array consisting of an array of red, green and blue values.

The loss function mean squared error works by summing together all the squared residuals and the gradient descent algorithm works with recording the errors and finding the lowest error corresponding to the linear function of the weight and bias [9].

GANs not only make it possible to generate faces, but also edit them. This is done by taking the base face image and using another face image to modify it. For instance in [10] an image of Barrack Obama being modified with another face which changes the skin texture for Obama to become slightly lighter in color and smoother than the original. The type of GAN that allows this is called a Style-GAN [11]. The same concept may be applied to other images such as animals, replacing images with horses with zebras or a summer landscape with winter, or colorizing black and white images [13].

One style of generating images using machine learning is by the use of auto encoding. This is a type of image compression which utilizes unsupervised learning as the data is learned without the use of labels [12].

According to [13] GANs go in the category of unsupervised learning as the structure of unlabelled data is learned as opposed to supervised learning in which the machine learns to fit a function to labelled data. For probability, the machine learns to predict the probability of outcome ‘y’ given the input ‘x’ (p(y | x)).

The log loss function is impractical for GANs due to their tendency to tarnish chances for low probability values such that they are given almost no chance to apply to the resulting function [13].

Perspective is a large issue in GANs as the networks don’t necessarily know the difference between a front and a back view [14].

An issue with using a GAN is what is called translation invariance. The GAN has to be able to learn to recognize objects regardless of their pixel position within the image. Whether the face is at the top corner or bottom, the GAN must be able to recognize the face itself and that alone is plenty of work for the network. Even though the faces or any other object the network learns often will all look similar to one another. Fortunately however, the solution to this issue is through weight sharing. This means that if the network notices that multiple images of faces have a similar structure, the same weights may be trained for the inputs combined. For the network to filter out as much possible of the background, by using convolutions to extract specific features of the image that matter and then combine them together known as pooling [15].

According to [17], SRGAN is proven to work also for face super resolution as well as any other application of super resolution.

## Methodology

## Development
For development of the application, the Kanban agile method will be used to monitor the progress and planning of the app. GitHub provides a customizable Kanban system for every project created. Test driven development (TDD) would be useful for the app if the app would need an API of its own, however since TensorFlow covers such an overwhelming amount of the machine learning operations, TDD should be less needed for this project. Scrum would be useful in a team which would include the scum master, however this is a solo project which makes Scrum more tedious in this scenario as opposed to its more scalable counterpart Kanban.
App
The idea of the neural network itself is not complex and neither will be in this scenario. A neural network simply may be thought of as a mathematical function with an extensive number of coefficients which change over time to match with as many possible conditions. The challenge in constructing the neural network mostly will take place in feeding training data and particularly the structure, which for faces will be useful to use CNNs so that features of the image is accounted for more so than the pattern of pixels that the image consists of. CNNs may use corresponding filters to detect features of a face, using e.g. horizontal, vertical, curved lines etc. to detect the jaw, the forehead, the cheekbones, eyes, nose etc. The activation function must be able to take the given value and adjust it to be within a valid range to ease the training process, by ultimately feeding a clean output to the loss function.

An important consideration to take in training the network is the training set(s). The face from the outside is composed of skin, nose, lips, eyes, eyelashes, eyebrows, ears, chin, hair, facial hair. There will need to be training sets for each of these features in high resolution. Training sets tend to be around 10k different samples, and these samples will need to have an even amount of training for people of different ethnicities across the world, which could only increase the difficulty in constructing this model. This network may be constrained to a clear face with no additional features such as tattoos and rings as simply a normal face alone will be complex. For gathering 10k HR face images, another neural network will be required to recognize face images from Google search results as the results are not guaranteed to display face images. Going this way will make it easier to gather a specified resolution for each face whilst also getting the exact face images required, whether it be the whole face or the parts. This would ease the process of gathering a higher resolution version of a basic face dataset.

For setting up a GAN, the first step is to have a set of noise images along with a set of real images with both sets being of the same size. As the network trains, the noise images will be mutated in the direction of resembling images from the real set.

## Outcomes

Future Development
For future development, more advanced faces will be possible to reconstruct such as faces with tattoos and/or rings and piercings. Alongside this, will be training for more ethnicities to include as many groups of people in the world, as after all exclusion of people of different races and ethnicities is an unfortunate issue in AI.

## References
[1] https://www.youtube.com/watch?v=KULkSwLk62I&t=141s

[2] https://www.youtube.com/watch?v=ppsqbOL073U

[3] https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf

[4] https://arxiv.org/abs/1705.02438

[5] https://www.coursera.org/learn/introduction-tensorflow/lecture/JSKji/what-are-convolutions-and-pooling

[6] https://zpascal.net/cvpr2018/Chen_FSRNet_End-to-End_Learning_CVPR_2018_paper.pdf

[7] https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/

[8] https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Z8j0R/binary-classification

[9] https://www.youtube.com/watch?v=sDv4f4s2SB8

[10] https://www.youtube.com/watch?v=dCKbRCUyop8

[11] https://arxiv.org/abs/1812.04948

[12] https://www.youtube.com/watch?v=3-UDwk1U77s

[13] https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

[14] https://github.com/songboning/GAN/blob/master/Doc/Learning%20Generative%20Adversarial%20Networks.pdf page 68-70

[15] https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

[16] https://www.academia.edu/38760533/Machine_Learning_for_Absolute_Beginners_A_Plain_English_Introduction

[17] https://github.com/eriklindernoren/Keras-GAN/blob/master/README.md
