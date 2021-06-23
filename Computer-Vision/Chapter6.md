# Chapter 6 Practical Aspects of Image Classification





## Generating CAMs

Feature maps are intermediate activations that come after a convolution operation. Typcially, the shape of these activation maps is 'n-channels x height x width'. If we take the mean of all these activations, they show the hotspots of all the classes in the image. we need to figure out only those feature maps among 'n-channels' that are responsible for that class. 

For the convolution layer that generated these feature maps, we can compute its gradients with respect to the 'cat' class. Note that only those channels that are responsible for predicting 'cat' will have a high gradient. This means that we can use the gradient information to give weightage to each of 'n-channels' and obtain an activation map exclusively for 'cat'

step by step

1. Decide for which class you want to calculate the CAM and for which convolutional layer in the neural network you want to compute the CAM.
2. Calculate the activations arising from any convolutional layer - let's say the feature shape at a random convolution layer is 512 x 7 x 7.
3. Fetch the gradient values arising from this layer with respect to the class of interest. The output gradient shape is 256 x 512 x 3 x 3 
4. Compute the mean of the gradients within each output channel. The output shape is 512.
5. Calculate the weighted activation map - which is the multiplication of the 512 gradients means by the 512 activation channels. 
6. Compute the mean of the weighted activation map to fetch an output of the shape 7 x 7 .
7. Resize the weighted activation map outputs to fetch an image of a size that is of the same size as the input. This is done so that we have an activation map that resembles the original image.
8. Overlay the weighted activation map onto the input image.

- If a certain pixel is important, the the CNN will have a large activation at those pixels.
- If a certain convolutional channel is important with respect to the required class, the gradients at that channel will be very large.z



![image-20210330124537303](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210330124537303.png)

**from ; Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**

- Fetch the convolution layer in the fifth 'convBlock' in the model.
  - fetching the fourth layer of the model and also the first two layers within 'convBlock' - which happens to be the 'Conv2D' layer.
- Define the 'im2gradCAM' function that takes an input image and fetches the heatmap corresponding to activations of image.
- Define the 'upsampleHeatmap' function to up-sample the heatmap to a shape that corresponds to the shape of the image.
  - De-normalizing the image and also overlaying the heatmap on top of the image.

## Understanding the impact of data augmentation and batch normalization

One clever way of improving the accuracy of models is by leveraging data augmentation. 

In the real world, you would encounter images that have different properties - for example, some images might be much brighter, some might contain objects of interest near the edges, and some images might be more jittery than others.

To understand the impact of data augmentation and batch normalization, 

- No batch normalization/data augmentation
- Only batch normalization, but no data augmentation
- Both batch normalization and data augmentation



## Coding up road sign detection



- The model did not have as high accuracy when there was no batch normalization.
- The accuracy of the model increased considerably but also the model overfitted on training data when we had batch normalization only but no data augmentation.
- The model with both batch normalization and data augmentation had high accuracy and minimal overfitting.



## Practical aspects to take care of during model implementation



- Dealing with imbalanced data.

rarely dataset , task scenario , 1% of the total images. For example, this can be the task of predicting whether an X-ray image suggests a rare lung infection.-> **accuracy is useless but Confusion Matrix is useful metrics at this scenario**

The loss function takes care of ensuring that the loss values are high when the amount of misclassification is high. However, in addition to the loss function, we can also assign a higher weight to the rarely occurring class, thereby ensuring that we explicitly mention to the model that we want to correctly classify the rare class images.

assigning class weights, we have already seen that image augmentation and/or transfer learning help considerably in improving the accuracy of the model. Furthermore, when augmenting an image, we can over-sample the rare class images to increase their mix in the overall population.

- The size of an object within an image when performing classification.

Imagine a scenario where the presence of a small patch within a large image dictates the class of the image - for example, lung infection identification where the presence of certain tiny nodules indicates an incident of the disease. ; image classification is likely to result in inaccurate results, as the object occupies a smaller portion of the entire image. 

A high-level intuition to solve these problems would be to first divide the input images into smaller grid cells and then identify whether a grid cell contains the object of interest.

- Dealing with the difference between training and validation images.

Imagine a scenario where you have built a model to predict whether the image of an eye indicates that the person is likely to be suffering from diabetic retinopathy. To build the model, you have collected data, curated it, cropped it, normalized it, and then finally built a model that has very high accuracy on validation images. **However, hypothetically, when the model is used in a real setting(let's say by a docetor/nurse), the model is not able to predict well.**

The reasons why a not well the case?!

- Are the images taken at the doctor's office similar to the images used to train the model?
  - Images used when training and real-world images could be very different if you built a model on a curated set of data that has all the preprocessing done, while the images taken at the doctor's end are non-curated.
  - Images could be different if the device used to capture images at the doctor's office has a different resolution of capturing images when compared to the device used to collect images that are used for training.
  - Images can be different if there are different lighting conditions at which the images are getting captured in both places
- Are the subjects representative enough of the overall population?
  - Images are representative if they are trained on images of the male population but are tested on the female population, or if, in general, **the training and real-world images correspond to different demographics.**
- Is the training and validation split done methodically?
  - Imagine a scenario where there are 10,000 images and the first 5,000 images belong to one class and the last 5,000 images belong to another class. When building a model, if we do not randomize but split the dataset into training and validation with consecutive indices, we are likely to see a higher representation of one class while training and of the other class during validation.

- The number of convolutional and pooling layers in a network.

In general, it is good practice to have a pre-trained model that obtains the flatten layer so that relevant filters are activated as appropriate. Furthermore, when leveraging pre-trained models, make sure to freeze the parameters of the pre-trained model.

- Image sizes to train on GPUs.

Let's say we are working on images that are of very high dimensions - for example, 2,000 x 1,000 in shape. When working on such large images, we need to consider the following possibilities;

** Can the images be resized to lower dimensions? Images of objects might not lose information if resized; however, images of text documents might lose considerable information if resized to a smaller size.

** Can we have a lower batch sized so that the batch fits into GPU memory? Typcially, if we are working with large images, there is a good chance that for the given batch size, the GPU memory is not sufficient to perform computations on the batch of images.

** Do certain portions of the image contain the majority of the information, and hence can the rest of the image be cropped?

- Leveraging OpenCV utilities.

OpenCV has been built on top of multiple hand-engineered features and at the time of writing this book, OpenCV has a few packages that integrate deep learning models' outputs.

Imagine a scenario where you have to move a model to production; less complexity is generally preferable in such a scenario - sometimes even at the cost of accuracy. If any OpenCV module solves the problem that you are already trying to solve, in general, it should be preferred over building a model (unless building a model from scratch gives a considerable boost in accuracy than leveraging off-the-shelf modules.)

## Summary

we learned about multiple practical aspects that we need to take into consideration when building CNN models - batch normalization, data augmentation, explaining the outcomes using CAMs, and some scenarios that you need to be aware of when moving a model to production.

## Insight

- Dashboard code

```python

log = Report(n_epochs)
for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model, data, optimizer, criterion)
        log.record(ex+(bx+1)/N, trn_loss=loss, trn_acc=acc, end='\r')

    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss, acc = validate_batch(model, data, criterion)
        log.record(ex+(bx+1)/N, val_loss=loss, val_acc=acc, end='\r')
        
    log.report_avgs(ex+1)
    if ex == 10: optimizer = optim.Adam(model.parameters(), lr=1e-4)

log.plot_epochs()
dumpdill(log, 'no-aug-yes-bn.log')
```





## Question