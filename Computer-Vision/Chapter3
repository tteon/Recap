# Chapter 3. Building a Deep Neural Network with PyTorch

We will cover the following topics;

- Representing an image
- Why leverage neural networks for image analysis
- Preparing data for image classification
- Training a neural network
- Scaling a dataset to improve model accuracy
- Understanding the impact of varying the batch size
- Understanding the impact of varying the loss optimizer
- Understanding the impact of varying the learning rate
- Understanding the impact of learning rate annealing
- Building a deeper neural network
- Understanding the impact of batch normalization
- The concept of overfitting

## Representing an image

An image has [ height x width x c ] pixels, where "height" is the number of rows of pixels, "width" is the number of columns of pixels, and "c" is the number of channels, c is 3 for color images ( one channel each for the red, green, and blue intensities of the image) and 1 for grayscale images.



## Converting images into structured arrays and scalars

- Histogram feature
  - auto-brightness or night vision
- Edges and Corners feature
  - image segementation , image registration
- Color separation feature
  - traffic light detection
- Image gradients feature
  - finding gradients is a prerequistie for edge detection

These are just a handful of such featrues. There are so many more that it is difficult to cover all of them. The main drawback of creating these features is that "you need to be an expert in image and signal analysis and should fully understand what features are best suited to solve a problem".

Due to these drawbacks, the community has largely shifted to neural network-based models. These models not only find the right features automatically but also learn how to optimally combine them to get the job don.

## Preparing our data for image classification

1. Start by downloading the dataset and importing the relevant packages.
2. Inspect the tensors that we are dealing with;
3. Plot a random sample of 10 images for all the 10 possible classes;

## Training a neural network

1. Import the relevant packages.

2. Build a dataset that can fetch data one data point at a time

   - Remember that it is derived from a 'Dataset' class and need  three magic function 1) init 2) getitem 3) len

3. Wrap the DataLoader from the dataset.

4. Build a model and then define the loss function and the optimizer.

5. Define two functions to train and validate a batch of data, respectively.

6. Define a function that will calculate the accuracy of the data.

   @ is decorator which disable the gradient computation in entire function ! 

7. Perform weight updates based on each batch of data over increasing epochs.

## Scaling a dataset to improve model accuracy

Scaling a dataset is the process of ensuring that the variables are confined to a finite range. In this section, we will confine the independent variables values to values between 0 and 1 by dividing each input value by the maximum possible value in the dataset. This is a value of 255, which corresponds to white pixels;

* just different between fetch function ' data is divided into 255 

why scaling helps here

-> how a sigmoid value is calculated ; sigmoid = 1 / (1 + e ^-(Input*Weight))

-> The sigmoid value does not vary with an increasing weight value.

Furthermore, the Sigmoid value vchanged only by a little when the weight was extremely small

Tip ; Scaling the input dataset so that it contains a much smaller range of values generally helps in achieving better model accuracy.

## Understanding the impact of varying the batch size

32 data points were considered per batch in the training dataset. This resulted in a greater number of weight updates per epoch as there were 1,875 weight updates per epoch

Furthermore, we did not consider the model's performance on an unseen dataset. we will explore this in this section.

- The loss and accuracy values of the training and validation data when the training batch size is 32.
- The loss and accuracy values of the training and validation data when the training batch size is 10,000.



we can see that the accuracy and loss values did not reach the same levels as that of the previous scenario, where the batch size was 32, because the time weights are updated fewer times when the batch size is 32. 

Tip ; Having a lower batch size generally helps in achieving optimal accuracy when you have a small number of epochs, but it should not be so low that training time is impacted.

## Understanding the impact of varying the loss optimizer

- Modify the optimizer so that it becomes a Stochastic Gradient Descent(SGD) optimizer
- Revert to a batch size of 32 while fetching data in the DataLoader
- Increase the number of epochs to 10(so that we can compare the performance of SGD and Adam over a longer number of epochs)

Tip ; Certain optimizers achieve optimal accuracy faster compared to others. Adam generally achieves optimal accuracy faster. some of the other prominent optimizers that are available include Adagrad, Adadelta, AdamW, LBFGS, and RMSprop

## Understanding the impact of varying the learning rate

The learning rate plays a key role in attaining optimal weight values. Here, the weight values gradually move toward the optimal value when the learning rate is small, while the weight value oscillates at a non-optimal value when the learning rate is large.

- Higher learning rate(0.1) on a scaled dataset
- Lower learning rate(0.00001) on a scaled dataset
- Lower learning rate(0.001) on a non-scaled dataset
- Higher learning rate(0.1) on a non-scaled dataset

## Impact of the learning rate on a scaled dataset

- High learning rate
- Medium learning rate
- Low learning rate

Tunning the hyper parameter 'learning rate' at optimizer section

### High learning rate

### Medium learning rate

### Low learning rate



## Parameter distribution across layers for different learning rates



- Weights in the layer connecting the input layer to the hidden layer
- Bias in the hidden layer
- Weights in the layer connecting the hidden layer to the output layer
- Bias in the output layer



- When the learning rate is high, parameters have a much larger distribution compared to medium and low learning rates.
- When parameters have a bigger distribution, overfitting occurs.

| ![image-20210309195149591](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210309195149591.png) | ![image-20210309195038819](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210309195038819.png) | ![image-20210309195402958](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210309195402958.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              | left high , middle ,  low learning rate                      |                                                              |



## Impact of varying the learning rate on a non-scaled dataset



The weights can be tuned toward a small value since the learning rate is small. Note that the scenario where the learning rate is 0.0001 on a non-scaled dataset is equivalent to the scenario of the learning rate being 0.001 on a scaled dataset. This is because the weights can now move toward a very small value

Tip ; Generally, a learning rate of 0.001 works. Having a very low learning rate means it will take a long time to train the model, while having a high learning rate results in the model becoming unstable.

## Understanding the impact of learning rate annealing

One potential way we can solve this problem is by continually monitoring the validation loss and if the validation loss does not decrease, then we reduce the learning rate.

PyTorch provides us with tools we can use to perform learning rate reduction when the validation loss does not decrease in the previous 'x' epochs. Here, we can use the 'lr_scheduler' method;



```python
from torch import optim
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, threshold = 0.001, verbose=True, min_lr = 1e-5, threshold_mode = 'abs')
```

Reducing the learning rate parameter of 'optimizer' by a 'factor' of 0.5 if a certain value does not improve over the next 'n' epochs (where n is 0 in this case) by a 'threshold'(which in this case is 0.001). Finally, we are specifying that the learning rate, 'min_lr'(given that it is reducing by a factor of 0.5),cannot be below 1e-5 and that 'threshold_mode' should be absolute to ensure that a minimum threshold of 0.001 is crossed.

## Building a deeper neural network

Contrast the performance of models where there are two hidden layers and no hidden layer (with no hidden layer being a logistic regression).

- The model was unable to learn as well as when there were no hidden layers.
- The model overfit by a larger amount when there were two hidden layers compared to one hidden layer.

## Understanding the impact of batch normalization

When the input value is very small, the Sigmoid output changes slightly, making a big change to the weight value.

Additionally, in the 'scaling the input data section', we saw that large input values have a negative effect on training accuracy. This suggest that we can neither have very small nor very big values for our input.

Along with very small or very big values in input, we may also encounter a scenario where the value of one of the nodes in the hidden layer could result in either a very small number or a very large number, resulting in the same issue we saw previously with the weights connecting the hidden layer to the next layer.

Batch normalization comes to the rescue in such a scenario since it normalizes the values at each node, just like when we scaled our input values.

Typically, all the input values in a batch are scaled as follows;
$$
Batch\ mean\ \mu_B = \frac1m \sum_{i=1}^{m}x_i
$$

$$
Batch \ Variance \ \sigma_2^{B} = \frac1m\sum_{i=1}^{m}(x_i - \mu_B)^2
$$

$$
Normalized \ input \ \bar{x_i} = \frac{(x_i-\mu_B)}{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
Batch \ normalized \ input = \gamma\bar{x}_i + \beta
$$



By subtracting each data point from the batch mean and then dividing it by the batch variance, we have normalized all the data points of the batch at a node to a fixed range.

While this is known as hard normalization, by introducing the \gamma and \beta parameters, we are letting the network identify the best normalization parameters.

- Very small input  values without batch normalization
- Very small input values with batch normalization

## Very small input values without batch normalization

## Very small input values with batch normalization



The hidden layer values have a larger distribution when we have batch normalization and that the weights connecting the hidden layer to the output layer have a smaller distribution. The results in the model elarning as effectively as it could in the previous sections.

Tip ; Batch normalization helps considerably when training deep neural networks. It helps us avoid gradients becoming so small that the weights are barely updated.



## The concept of overfitting

- Dropout
- Regularization



' when the same model is in eval mode, it will suppress the dropout layer and return the same output'

## Impact of regularization

- L1 regularization
- L2 regularization

### L1 regularization

$$
L1 \ loss \ = -\frac{1}{n}(\sum_{i=1}^{n}(y_i*log(p_i) + (1 - y_i) * log(1 - p_i)) + \land\sum_{j=1}^{m}|w_j|)
$$

The first part of the preceding formula refers to the categorical cross-entropy loss that we have been using for optimization so far, while the second part refers to the absolute sum of the weight values of the model.



Note that L1 regularization ensures that it penalizes for the ghigh absolute values of weight by incorporating them in the loss value calculaton.

\land refers to the weightage that we associate with the regularization (weight minimization) loss.

```python
def train_batch(x, y, model, opt, loss_fn):
	model.train()
	prediction = model(x)
	l1_regularization = 0 ##
	for param in model.parameters():
		l1_regularization += torch.norm(param,1)
	batch_loss = loss_fn(prediction, y)+0.0001*l1_regularization ##
	batch_loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	return batch_loss.item()
```

​	Enforcing regularization on the weights and biases across all the layers by intializing 'l1_regularization'

'torch.norm(param,1)' provides the absoulte value of the weight and bias values across layers.

### L2 regularization

$$
L2loss = -\frac{1}{n}(\sum_{i=1}^{n}(y_i*log(p_i) + (1 - y_i) *log(1 - p_i)) + \land\sum_{j=i}^mw^2_j
$$

```python
def train_batch(x, y, model, opt, loss_fn):
	model.train()
	prediction = model(x)
	l2_regularization = 0 ##
	for param in model.parameters():
		l1_regularization += torch.norm(param,2)
	batch_loss = loss_fn(prediction, y) + 0.01*l2_regularization ##
	batch_loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	return batch_loss.item()
```



## Summary

learning about how an image is represented. Next, we learned about how scalihng, the value of the learning rate, our choice of optimizer, and the batch size help improve the accuracy and speed of training. We then learned about how batch normalization helps in increasing the speed of training and addresses the issues of very small or large values in hidden layer. Next, we learned about scheduling the learning rate to increase accuracy further. We then proceeded to understand the concept of overfitting and learned about how dropout and L1 and L2 regularization help us avoid overfitting.



## Question

1. What happens if the input values are not scaled in the input dataset?
2. What could happen if the background has a white pixel color while the content has a black pixel color when you're training a neural network?
3. What impact does the batch size have on the model's training time, as well as its accuracy over a given number of epochs?
4. What impact does the input value range have on the weight distribution at the end of the training?
5. How does batch normalization help improve accuracy?
6. How do we know if a model has overfitted on training data?
7. How does regularization help in avoiding overfitting?
8. How do L1 and L2 regularization differ?
9. How does dropout help in reduction overfitting?
