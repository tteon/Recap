# Chapter 4



## part1

- The problem with traditional deep neural networks 

1. Fetch a random image from the available training images
2. Pass the image through the trained model
3. Translate the image multiple times from a translation of 5 pixels to the left to 5 pixels to the right and store the predictions in a list 
4. Visualize the predictions of the model for all the translations

The predicted class of the image changed when the translation was beyond 2 pixels. This is because while the model was being trained, the content in all the training and testing images was at the center. This differs from the preceding scenario where we tested with translated images that are off-center, resulting in an incorrectly predicted class.

- Building blocks of a CNN

  - Convolution

  A convolution is basically multiplication between two matrices. 'Matrix multiplication is a key ingredient of training a neural network.'

  - Filter

  A filter is a matrix of weights that is initialized randomly at the start. The model learns the optimal weight values of a filter over increasing epochs.

  The concept of filters brings us to two different aspects;

  1. What the filter learn about
  2. How filters are represented

  In general, the more filters there are in a CNN, the more features of an image that the model can learn about. We will learn about what various filters learn in the 'Visualizing the filters' learning section of this chapter.

  In terms of height and width. This results in a partial loss of information and can affect the possibility of us adding the output of the convolution operation to the original image.

  - Strides and padding
    - Strides

    

    original

    | 44   | 54   | 64   |
    | ---- | ---- | ---- |
    | 84   | 94   | 104  |
    | 124  | 134  | 144  |

    stride 1 adaption

    | 44   | 64   |
    | ---- | ---- |
    | 124  | 144  |

    

    

    - padding

    This would ensure that we can perform element to element multiplication of all the elements within an image with a filter

  - Pooling

  Pooling aggregates information in a small patch.

- Putting them all together

the flatten layer (fully connected layer) - before putting the three pieces we have learned about together.

The operations of convolution and pooling constitute the feature learning section as filters help in extracting relevant features from images and pooling helps in aggregating information and thereby reducing the number of nodes at the flatten layer.

Convolution and pooling help in fetching a flattened layer that has a much smaller representation that the original image.

The classification is similar to the way we classified images in 'Chapter 3'

## part2

- How convolution and pooling help in image translation

we'll have reduced the dimension of the image(due to pooling), which means that a fewer number of pixels store the majority of the information from the original image. Moreover, given that pooling stores information of a region (path), the information within a pixel of the pooled image would not vary, even if the original image is translated by 1 unit. This is because the maximum value of that region is likely to get captured in the pooled image.

Convolution and pooling cam also help us with the 'receptive field'. To understand the receptive field, let's imagine a scenario where we perform a convolution pooling operation twice on an image that is 100 x 100 in shape. The output at the end of the two convolution pooling operations is of the shape 25 x 25 (if the convolution operation was done with padding). Each cell in the 25 x 25 output now corresponds to a large 4 x 4 portion of the original image. Thus, because of the convolution and pooling operations, each cell in the resulting image corresponds to a patch of the original image.

- Implementing a CNN

  - Convolution operation
  - Pooling operation
  - Flattening layer

1. need to import the relevant libraries;
2. Create the dataset 
3. Constructing the model architecture
4. Summarize the model blueprint

** note that PyTorch expects inputs to be of the shape _N x C x H x W_, where N is the number(batch size) of images, C is the number of channels, H is the height, and W is the width of the image.

5. Training the model
6. Perform a forward pass on top of the first data point

Note that you might have a different output value owing to a different random weight initialization when you execute the preceding code. However, you should be able to match the output against what you get in the next section.



- Building a CNN-based architecture using PyTorch



- Forward propagating the output in Python

this section is only here to help you clearly understand how CNNs work.

1. Extract the weights and biases of the convolution and linear layers of the architecture 
2. To perform the 'cnn_w' convolution operation over the input value, we must initialize a matrix of zeros for sumproduct.
3. let's fill 'sumprod' by convolving the filter across the first input and summing up the filter bias term after reshaping the filter shape from a 1 x 1 x 3 x 3 shape to a 3 x 3 shape;
4. Perform the ReLU oepration on top of the output and then fetch the maximum value of the pool
   1. ReLU is performmed on top of 'sumprod' .
   2. The output of the pooling layer can be calculated.
5. Pass the preceding output through linear activation.
6. Pass the output through the 'sigmoid' operation.

- Classifying images using deep CNNs

In this section, Understand how CNNs address the problem of incorrect predictions when image translation happens on images in the Fashion-MNIST dataset.

1. packages

2. 'Dataset' object will **always** need the init , getitem , len methods 

    input ; batch size x channels x height x width

3. CNN model 

![image-20210311192443272](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210311192443272.png)

- Layer 1 ; Given that there are 64 filters with a kernel size of 3, we have 64 x 3 x 3 weights and 64 x 1 biases, resulting in a total of 640 parameters.
- Layer 4 ; Given that there are 128 filters with a kernel size of 3, we have 128 x 64 x 3 x 3 weights and 128 x 1 biases, resulting in a total of 73,856 parameters.
- Layer 8 ; Given that a layer with 3,200 nodes is getting connected to another layer with 256 nodes, we have a total of 3,200 x 256 weights + 256 biases, resulting in a total of 819,456 parameters.
- Layer 10 ; Given that a layer with 256 nodes is getting connected to a layer with 10 nodes, we have a total of 256 x 10 weights and 10 biases , resulting in a total of 2,570 parameters.

As we can see, while CNNs help in addressing the challenge of image translation, they don't solve the problem at hand completely. 

## part3

- Implementing data augmentation

The same image will be processed as a different image in different passes since it will have had a different amount of translation in each pass.

- Image augmentations

! image augmentations come in handy in scenarios where we create more images from a given image. Each of the created images can vary in terms of rotation, translation, scale, noise, and brightness. Furthermore, the extent of the variation in each of these parameters can also vary

The 'augmenters' class in the 'imageaug' package has useful utilities present in the 'augmenters' class for generating augmented images from a given image.

! Note that PyTorch has a handy image augmentation pipeline in the form of 'torchvision.transforms'. However, we still opted to introduce a different library primarily because of the larger variety of options 'imgaug' contains, as well as due to the ease of explaining augmentations to a new user. You are encouraged to research the torchvision transforms as an exercise and recreate all the functions that are presented to strengthen your understanding.

1. Affine transformations

'Affine'method

- 'scale' specifies the amount of zoom that is to be done for the image
- 'translate_percent' specifies the amount of translation as a percentage of the image's height and width
- 'translate_px' specifies the amount of translation as an absolute number of pixels
- 'rotate' specifies the amount of rotation that is to be done on the image
- 'shear' specifies the amount of rotation that is to be done on part of the image



! https://uos-deep-learning.tistory.com/17 is explain the technique 'imgaug'

1. Change brightness

   1. Multiply

   Multiplies each pixel value by the value that we specify. The output of multiplying each pixel value by 0.5 for the image 

   1. Linearcontrast

   $$
   127 + \alpha \ \times (pixelvalue - 127)
   $$

   in the preceding equation, when \alpha is equal to 1, the pixel values remain unchanged. However, when \alpha is less than 1 , high pixel values are reduced and low pixel values are increased.

   above two functions can be leveraged to resolve such scenarios.

   + GaussianBlur

2. Add noise

   1. Dropout

   Dropped a certain amount of pixels randomly(that is, it converted them so that they had a pixel value of 0)

   1. SaltAndPepper

   Added some white-ish and black-ish pixels randomly to our image.

- Performing a sequence of augmentations

Sequential way of performing augmentations.

```python
seq = iaa.Sequential([
iaa.Dropout(p=0.2),
iaa.Affine(rotate=(-30,30))], random_order=True)

```

- Performing data augmentation on a batch of images and the need for collate_fn

getitem method - which is ideal since we want to perform a different set of augementations on each image -

### !! collate_fn that enables us to perform manipulation on a batch of images !!

- Data augmentation for image translation

## part4

- Visualizing the outcome of feature learning

glob는 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 입맛대로 요리할 수 있답니다.

1. Download the dataset
2. Import the required modules
3. Define a class that fetches data , ensure that #1 the images have been resized to a shape of 28 x 28, #2 batches have been shaped with three channels, and that #3 the dependent variable is fetched as a numeric value. ☆
4. Inspect a sample of the images you've obtained. Extracting the images and their corresponding classes by fetching data from the class we defined
5. Define the model architecture, loss function, and the optimizer
6. Define a function for training on batches that takes images and their classes as input and returns their loss values and accuracy after backpropagation has been performed on top of the given batch of data
7. Define a 'DataLoader' where the input is the 'Dataset' class
8. Initialize the model
9. Train the model over '5' epochs
10. Fetch an image to check what the filters learn about the image
11. Pass the image through the trained moel and fetch the output of the first layer. Then, store it in the 'intermediate_output' variable
12. Plot the output of the 64 filters. Each channel in 'intermediate_output' is the output of the convolution for each filter
13. Pass multiple O images and inspect the output of the fourth filter across the images
14. Plot the output of passing multiple images through the 'first_layer'model
15. Create another model that extract layers until the second convolution layer and then extracts the ouptut of passing the original O image. We will then plot the output of convolving the filters in the second layer with the input O image
16. Plot the activations of a fully connect layer

- Building a CNN for classifying real-world images

1. library
2. dataset
3. train and test dataset
4. Folder , ensure that the fetched image has been normalized to a scale between 0 and 1 and permute if so that channels are provided first 
5. Inspect a random image
6. Define model , loss function , optimizer
7. get_data function , cats_dogs class and creates a DataLoader with a batchsize of 32 for both the training and validation folders
8. function that will train the model on a batch of data 
9. functions for calculating accuracy and validation loss
10. Train the model for 5 epochs and check the accuracy of the test data at the end of each epoch
11. Plot the variation of the training and validation accuracies 

Tip ;  batch normalization has a great impact on improving classification accuracy ; you can do this by reducing the number of layers, increasing the stride, increasing the pooling, or resizing the image to a number that's lower than 224 x 224

- Impact on the number of images used for training

dataset[:500]

- Summary

Traditional neural newtorks fail when new images that are very similar to previously seen images that have been translated are fed as input to the model. CNN play a key role in addressing this shortcoming. This is enabled through the various mechanisms that are present in CNNs, including filters, strides, and pooling. 

we learned about how data augmentation helps in increasing the accuracy of the model by creating translated augmentations on top of the original image and what different filters learn in the feature learning process so that we could implement a CNN to classify images.

impact that differing amounts of training data have on the accuracy of test data.



## Question

1. Why is the prediction on a translated image low when using traditional neural networks? 
2. How is convolution done? 
3. How are optimal weight values in a filter identified?
4. How does the combination of convolution and pooling help in addressing the issue of image translation?
5. What do the filters in layers closer to the input layer learn?
6. What functionality does pooling have that helps in building a model?
7. Why can't we take an input image, flatten it, and then train a model for real-world images? ## 
8. How does data augmentation help in improving image translation?
9. In what scenario do we leverage 'collate_fn' for DataLoader?
10. What impact does varying the number of training data points have on the classification accuracy of the validation dataset?



