# TextRecognition Project, by Emily Beatty, completed 1/6/21

## RUNNING INSTRUCTIONS:

1. Open terminal

2. Navigate to /home/learner/EmilysProject/TextRecognition

3. Run main python file instructions one at a time on the command line: *I suggest executing only i and ii*. I have provided iii, iv, and v if you would like to run those as well.

    1. To run my pre-trained model with the MNIST Test data do: `python3 Main.py 1`

    2. To run my pre-trained model with my handwritten Test data do: `python3 Main.py 2`

    3. To retrain model and evaluate with MNIST Test data do:`python3 Main.py 3`

    4. To run Kfold cross validation and see results do (NOTE: this one takes a while to execute): `python3 Main.py 4`

    5. To create and save bar chart for category distribution do: `python3 Main.py 5`

## GITHUB

The repo is in Github and is a private repo as of now. For most of the development, Jupyter Notebook was used, so all development commits are focused on the Main.ipynb file. The file was converted to a regular python file (Main.py) so that it could be executed from the commandline. I will provide access to the Technical Point of Contact through email.  

## REPORT

### Introduction:

I decided to do a mini project on handwriting recognition for this interview. I recently took a class on Computer Vision (CV) and was inspired to do a project that involved images. In CV, I learned about traditional methods to interpret information in images using techniques like Haar features, SVM, KNN, and boosted trees. Although these methods work well, many students in my class made it clear that some of these problems could be better solved using deep learning. I was excited to explore this idea further. 

I was first introduced to the subject as an intern, where I worked in a group to stand up a conventional optical character recognition (OCR) engine for old government documents that needed the SSN redacted. It worked well most of the time but was less accurate when the scanned in document was warped or distorted. I did not think much about this problem again until I was in CV. I came across a paper that used the Distance Transform to find the white lines between text instead of the text itself [1]. The results from the paper were good - more text was captured from the documents, but still not great. There are many articles and research papers that suggest and show that Deep Learning can outperform conventional CV techniques for OCR, especially when images are distorted, warped, or reduced in quality. However, what about handwritten text?

About a month ago I was scrolling through LinkedIn and saw a post by Allie K. Miller, an employee at AWS. She showed off how Amazon was launching Textract, a handwriting recognition engine, with a short video of her writing a sentence and the engine pulling out each word [2]. From research, I know Deep Learning is also the answer (or at least the state of the art) for handwriting recognition [3]. The video stuck with me and I wanted to understand how handwriting recognition works. While I was not able to create my own Textract, I thought it would be interesting to understand how CNNs are used on the MNIST database to recognize handwritten numerical characters.

 
### Experiment:

There are many different datasets for handwritten character recognition. I investigated EMNIST and some Kaggle datasets before I decided to go with MNIST. I felt that numerical characters would be a good starting place and the dataset was conveniently sized so that training would not take too long [4].

There appears to be a few ways to load the MNIST dataset, but I found manually downloading the files, unzipping, and then using the mnist python library worked best with my environment. According to [4] the images are 28x28 are already grayscale images. The images where converted to float to preserve information and normalized to the range 0-1. At first, I did not understand why I would need to normalize the images, as [6] suggests. The images come in as unsigned uint8 with values between 0 and 255, a defined range. However, upon research in [5] and [7] I learned that normalizing between 0 and 1 mitigates the risk of disrupted or slowed down learning since weight values are typically small.

The implementation I went with is a supervised learning approach. The label arrays for training and test were also preprocessed. One hot encoding is a newer concept for me. I was introduced to it at work for an ELP project. The concept makes a lot of sense since the magnitude of the values in the labels array has no significance and it is instead a categorical problem. One hot encoding transforms the (-1, 1) array to a (-1, 10) array. Each row in the new array will contain a 1 at the column index that represents the numerical value and a 0 everywhere else. For example, if the value was 4, the 1 is placed at the 4th index (5th column) since we start from 0 for that row.

Next, I wanted to understand how balanced the dataset was. I had high expectations that the MNIST dataset was balanced, but I wanted to create a visual to see it. A quick way to do this is to count the labels for each category in the dataset and then display it in a bar chart. See saved bar charts for Train (Train\_barChart.png) and Test (Test\_barChart.png) data in the current directory.

I searched for an algorithm that I could apply to my dataset to determine whether the dataset was balanced or not. I could not find an industry standard, but I was able to pick up a rule of thumb from stack overflow - 50/50 split or 60/40 splits are acceptable. Applying this to a multiclass problem, I calculated the distribution of values in the data and found the category for number 1 to be the most represented with a percentage of 11.35% and the least represented to be the category number 5 with only 8.92% representation. Since there are 10 classes, the ideal distribution would be 10% for each class but it is still more balanced than a 60/40 split. The same calculations were performed on the test dataset provided by [4]. I concluded that the datasets were balanced enough to be considered Balanced.

Now that the datasets are ready for training, I investigated cross validation techniques. Currently the dataset is separated into training and test data. There are 60000 samples in train and 10000 in test. The split was provided by [4]. However, it can be dangerous to only use training and test to evaluate the model. According to [8] you can overfit the model such that it performs well on the test data due to repeated tuning and adjustments to get the model to predict the test data well. This is bad practice, and I have heard in industry that the customer will withhold the test data, so this does not happen.

Cross validation is a good solution as it solves a handful of problems. First, it allows you to create a validation set from the training data to test the accuracy while experimenting without touching the test data. Second, it allows you to test k number of training/validation sets, where k-1 is trained on while 1-fold is left out for validation. Third, it allows you to see the changes in accuracy and error for each iteration of training/validation splits. Furthermore, there are different types of cross validation techniques that exist. I investigated KFold and Stratified KFold. Stratified KFold caught my eye since it ensures that the same distribution of label values are in each fold. This could be especially important if the datasets are imbalanced. However, through my analysis above, I concluded that I have a balanced dataset, so I chose to go with KFold. KFold randomly selects datapoints with no concern for the label value. 

I chose to implement a simple Convolutional Neural Network that I found in [6]. I have read up on CNNs and how they are particularly good at object detection and recognition for image data. I spent my time understanding how each layer works.

 `Sequential()`   
 `Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')`   
 `MaxPooling2D((2, 2))`   
 `Flatten()`   
 `Dense(100, activation='relu', kernel_initializer='he_uniform')`   
 `Dense(10, activation='softmax')`   

The first layer in the CNN is the Conv2d layer. This layer constructs a kernel by sampling from a uniform distribution with a window size of 3. Sampling allows for randomness which is good for training accuracy [13]. To perform a convolution, which is really cross-correlation, is to slide the 3x3 kernel over the image from left to right. At each kernel location, the kernel is multiplied with pixels at the same location and the sum is stored in the result. Convolutions are used commonly in CV as a means of feature extraction. Although in CV, the kernel weight values are usually deliberate. For example, if I was looking for an "X" in an image, I might use a kernel like this:

[[1 -1 1],   
 [-1 1 -1],   
 [1 -1 1]]   


In this problem, the weights must be learned. I wanted to understand how the weights are learned and the purpose of 32 filters in the Conv2d.

Each filter performs a convolution over the image such that the output will have a depth of 32. However, instead of just extracting one feature map at a time, the Conv2d allows convolutions to run in parallel so that multiple features can be extracted. Since the Conv2d is the first layer, the input are the raw pixels themselves, so only lower-level features are extracted [12].  

The activation chosen is ReLu - a simple non-linear function. Anything less than 0 becomes 0 since the function is y = 0 for all negative x values. All positive values are left unchanged since the function becomes y = x when x is positive. The important thing about ReLu, and other activation functions in deep learning is that they are non-linear. Why does this matter? Well, without non-linearity there would only be linear transformation where the output would be linear. There would also be no need for layers if every layer were linear [9]. Furthermore, ReLu will help the model generalize better.

The next layer is max pooling. Pooling in some ways is like convolution, there is a 2x2 kernel that slides over each feature map. They differ in that in max pooling, the maximum value of the feature map in the 2x2 space is taken and placed in the result and that it is nonlinear. One benefit to this method is that it reduces the size of the output, while preserving the most distinct features. Another benefit is that it makes the model invariant to translation [14]. Translation invariance allows features to be moved up/down, left/right and the model will still be able to detect the feature. So, when the max value is taken, it changes where the precise location of where that feature was detected in the image, which causes the desired invariance [14].

The next layer is Flatten. The layer's function reminds me a lot of NumPy reshape function. It takes in an input (13, 13, 32), note that it is 13 and not 16 due to the convolution operation and the option to not add padding in conv2d layer, and flattens it to a 1D C-style array of size 13\*13\*32.

The last two layers are Dense layers. At a high level, they reduce the dimensionality from 5408 to 100 to 10. It is important that the model's output layer (last layer) end with only 10 neurons since there are 10 possible number categories the image could be. The softmax function is suitable as the activation in the last layer since it converts the vector values into probabilities that sum to 1. The highest probability category is what the model selects as the number that the image represents. Although, softmax is most important as it uses cross-entropy loss. The loss is calculated using the softmax probability for the correct category which will quantify how much confidence there is in the prediction [15].

So how is the model trained? The weights cannot continue to be completely random, or the model would never learn. The weights need to be updated using a technique called back propagation. The best way to think of this is by starting with the last layer. According to [16], backpropagation uses the Loss gradient, the correct category, and the weights gradient to update the weights. While I am not too familiar with the inner workings of backpropagation, I can speak about gradients. In CV, it is common to find the gradient of an image (partials of x and y) to find edges in an image. Like any derivative, the largest change in values, and in this case, pixel values, produces a large value derivative while small differences produce a small value. As you can imagine, edges are high contrast areas for pixel intensity. Translating this thinking to backpropagation, the Loss gradient finds the slope of loss. Steep slopes should indicate high values of Loss which by all accounts are bad. The model must update weights such that the Loss slope continues to be reduced as much as possible, where the ideal Loss gradient has a slope of 0.  


### Results:

The final CNN model was trained on the entire training set once cross validation results showed the data was balanced and the model was able to predict all categories similarly well (98% accuracy). The model was tested using the MNIST test set with an accuracy around 99%. The accuracy improved slightly from cross-validation to complete training due to using the entire training set. The MNIST dataset however is somewhat idealized, with little to no noise in images. The numbers also appear to be binarized and are slightly smoothed. The numbers in the images are also centered and scaled. The max pooling layer can help CNNs become invariant to some transformations, but with this simple example it is hard to say how invariant the model is to rotation, scale, and translation. I ended up creating my own test set to see how the model does. I wrote out numbers 0 through 9 and saved them as image files. I cropped the images to 28x28 but did not use any techniques to scale or center the numbers perfectly. I converted to grayscale and used a OpenCV inverse binarize function with a threshold to turn the paper black and the number white to mimic the MNIST dataset format. Without any other pre-processing, my model was able to predict 8 out of the 10 numbers correct. I would imagine further preprocessing, or a more robust model could increase the accuracy of the model on my handwritten dataset.

### Conclusion:

Handwriting recognition is a cool topic! I am glad I had the opportunity to dive into CNNs and how they can be applied to this problem. Although, there is still so much to learn. I would like to continue to explore backpropagation, layers, hyperparameters for each layer, and ways to visualize results as I have barely scratched the surface in this project.   

For example, when I was researching cross-validation, I could not find a standard for scoring techniques to use on multiclass problems. In my project, I only reported on the accuracy of each fold, but I know that using different scoring functions allows the consistency of the model to be analyzed. For example, if I have a binary dataset where there are eight 0s and two 1s and the model guess 0 for all samples, the accuracy will still show 80%, but does not reveal that the model only guesses 0.

Another example is the CNN. Using Keras was good and bad. It was good because there are a lot of resources on Keras CNNs, but bad because it is very high-level. Part of the mystery of deep learning is explaining the model and why it works which was harder to understand with Keras. I started to get a decent grasp on the feed-forward process in this project, but the backpropagation is still a work in progress for me. I understand the reason to use the gradient, but less certain on how the gradient transforms as it is passed back through the other layers and how the weights change based on the gradient. Furthermore, I found it hard to tinker with the hyperparameters in the layers partly due to Keras, but also not knowing what to change to improve the model. While my project did not require much tinkering, I still found myself thinking about what I would do had I needed to improve the model accuracy. There seems to be three major things that can be changed: data, model, hyperparameters. Perhaps the model has bad accuracy due to a small amount of data, imbalanced dataset, or quality of the data. On the other hand, the model chosen could just be bad at predicting that kind of data for the problem. Or better yet, the data is good, and the model selected is also good, but the hyperparameters need to be tuned.



##  REFERENCES:
[1]Distance Transform Paper, URL: https://www.cse.sc.edu/~songwang/document/wacv15b.pdf   
[2]Textract software, URL: https://aws.amazon.com/textract/   
[3]Image Recognition Based on Deep Learning, URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7382560   
[4]The MNIST Database of handwritten digits, URL: http://yann.lecun.com/exdb/mnist/   
[5]Why normalize images for deep learning, URL: https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current   
[6] How to develop a CNN for handwriting recognition, URL: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/   
[7] How to manually scale image pixel data for deep learning, URL: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/   
[8] Cross-Validation, URL: https://scikit-learn.org/stable/modules/cross\_validation.html   
[9] Why do you need non-linear activation functions? URL: https://www.coursera.org/lecture/neural-networks-deep-learning/why-do-you-need-non-linear-activation-functions-OASKH   
[10] Simple MNIST Convnet, URL: https://keras.io/examples/vision/mnist\_convnet/   
[11]Convolutional Neural Network, URL: https://towardsdatascience.com/convolutional-neural-network-ii-a11303f807dc   
[12] Keras Conv2d and Convolutional Layers, URL: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/   
[13] The effects of weight initialization on neural nets, URL: https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets   
[14] Pooling Layers, URL: https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/   
[15] Deep AI, URL: https://deepai.org/machine-learning-glossary-and-terms/softmax-layer   
[16] Cross-Entropy Loss, URL: https://victorzhou.com/blog/intro-to-cnns-part-1/#52-cross-entropy-loss   
