# Chapter 5 Transfer Learning for Image Classification



### part1

We will learn about optimizing over both cross-entropy and mean absolute error losses at the same time, leveraging neural networks to generate multiple continuous outputs in a single prediction. New library that assists in reducing code complexity considerably across the remaining chapters.

## Introducing transfer learning

1. Normalize the input images, normalized by the **same mean and standard deviation** that was used during the training of the pre-trained model.
2. fetch the pre-trained model's architecture. Fetch the weights for this architecture that arose as a result of being trained on a large dataset.
3. Discard the last few layers of the pre-trained model.
4. Connect the truncated pre-trained model to a freshly initialized layer(or layers) where weights are randomly initialized. Ensure that the output of the last layer has as many neurons as the classes/outputs we would want to predict
5. Ensure that the weights of the pre-trained model are not trainable, but that the weights of the newly initialized layer and the weights connecting it to the output layer are trainable;
   - We do not train the weights of the pre-trained model, as we assume those weights are already well learned for the task, and hence leverage the learning from a large model. In summary, we only learn the newly initialized layers for our small dataset.
6. Update the trainable parameters over increasing epochs to fit a model.

## Understanding VGG16 architecture

- **VGG** stands for **Visual Geometry Group**,

- We would freeze the 'features' and 'avgpool' modules. and the Delete the 'classifier' module (or noly a few layers at the bottom) and create a new one in its place that will predict the required number of classes corresponding to our dataset
- While levearging pre-trained models, it is mandatory to resize, permute, and then normalize images, where the images are first scaled to a value between 0 and 1 across the 3 channels and the nnormalized to a mean of ~ and a standard deviation of ~ 
- 'nn.MaxPool2d', where we are picking the maximum value from every section of a feature map. There is a counterpart to this layer called 'nn.AvgPool2d', which returns the average of a section instead of the maximum. In both these layers, we fix the kernel size. The layer above, 'nn.AdaptiveAvgPool2d', is yet another pooling layer with a twist. We specify the output feature map size instead. The layer automatically computes the kernel size so that the specified feature map size is returned.
  - For example, if the input feature map size dimensions were 'batch_size x 512 x k x k', then the pooling kernel size is going to be 'k x k'. The major advantage with this layer is that whatever the input size, the output from this layer is always fixed and, hence, the neural network can accept images of any height and width.
- Define the 'classifier' module of the model, where we first flatten the output of the 'avgpool' module, connect the 512 units to the '128'units, and perform an activation prior to connecting to the output layer;
- **we have first frozen all the parameters of the pre-trained model and have the overwritten the 'avgpool' and 'classifier' modules.**
- Note that the number of trainable parameters is only 65,793 out of a total of 14.7 million, as we have frozen the 'features' module and have overwritten the 'avgpool' and 'classifier' modules. Now, only the 'classifier' module will have weights that will be learned.

## Understanding ResNet architecture

While building too deep a network, there are two problems. In forward propagation, the last few layers of the network have almost no information about what the original image was. In back propagation, the first few layers near the input hardly get any gradient updates due to vanishing gradients.To solve both problems, residual networks use a highway-like connection that transfers raw information from the previous few layers to the layer layers.

The term 'residual' in the residual network is the additional information that the model is expected to learn from the previous layer that needs to be passed on to the next layer. [ extracting not only the value after passing through the weight layers, which is F(x), but are also summing up F(x) with the original value ]

-> This way , in certain scenarios, the layer has very little burden in remembering what the input is, and can focus on learning the correct transformation for the task.

### Building ResLayer

```python
class ResLayer(nn.Module):
    def __init__(self, ni, no, kernel_size, stride=1):
        super(ResLayer, self).__init__()
        padding = kernel_size - 2
        self.conv = nn.Sequential(
            nn.Conv2d(ni, no, kernel_size, stride, padding=padding),
            nn.ReLU()
        )

def forward(self, x):
    x = self.conv(x) + x
    return x        
```

Components

- Convolution
- Batch normalization
- ReLU
- MaxPooling
- Four layers of ResNet blocks
- Average pooling
- A fully connected layer

So far , we did binary classification , in next section learn about leveraging pre-trained models to solve real-world use cases 

- Multi-regression ; Prediction of multiple values given an images as input-facial key point detection
- Multi-task learning ; Prediction of multiple items in a single shot - age estimation and gender classification



### part2

## Implementing facial key point detection

- Image can be of different shapes;
  - This warrants an adjustment in the key point locations while adjusting images to bring them all to a standard image size.
- Facial key points are similar to points on a scatter plot, but scattered based on a certain pattern this time;
  - This means that the values are anywhere between 0 and 224 if the image is resized to a shape 224 x 224 x 3
- Normalize the dependent variable 
  - The key point values are always between 0 and 1 if we consider their location relative to image dimensions.
- Given that the dependent variable values are always between 0 and 1, we can use a sigmoid layer at the end to fetch values that will be between 0 and 1.

Sequence

1. Import the relevant packages.

2. Import data

3. Define the class that prepares the dataset;

   - Ensure appropriate pre-processing is done on input images to perform transfer learning.
   - Ensure that the location of key points is processed in such a way that we fetch their relative position with respect to the processed image.

4. Define the model, loss function, and optimizer;

   - The loss function is the MAE, as the output is a continuous value between 0 and 1.

5. Train the model over increasing epochs.

   - expected output will always be between 0 and 1 as keypoint locations are a fraction of the original image's dimensions;

   - L1Loss , MAE 

## 2D and 3D facial key point detection

leverage the 'face-alignment' library

- face-alignment , landmark , Plotly

1. packages
2. image
3. face alignment method , where we specify whether we ant to fetch key point landmarks in 2D or 3D
4. Read the input image and provide it
5. Plot the image with the detected key points

## Multi-task learning - Implementing age estimation and gender classification

- one task at a time
- learn about predicting both attributes, continuous and categorical predictions
  1. relevant packages
  2. Fetch a dataset that contains images of persons, their gender, and age information
  3. Create training and test datasets by performing appropriate pre-processing.
  4. Build a model where the following applies.
- In the last part, create two separate layers branching out from the preceding layer, where one layer corresponds to age estimation and the other to gender classification.
- Ensure that you have different loss functions for each branch of output, as age is a continuous value and gender is a categorical value
- Take a weighted summation of age estimation loss and gender classification loss.
- Minimize the overall loss by performing back-propagation that optimizes weight values.
  5. Train model and predict on new images.

## Introducing the torch_snippets library

A library that the authors have built to avoid such verbose code.

The 'Reopoprt' class is instantiated with the only argument, the number of epochs to be trained on, and is instantiated just before the start of training.

'Report.record' method with exactly one positional argument, which is the position of training/ validation we are at .

without creating a single empty list.



torch_snippets;

## Summary



## Question