

# Basics of Object Detection



topics

- Introducing object detection
- Creating a bounding box ground truth for training
- Understanding region proposals
- Understanding IoU, non-max suppression, and mean average precision
- *Training R-CNN-based custom object detectors*
- *Training Fast R-CNN-based custom object detectors*



**ground truth of bounding box objects using a tool named 'ybat'**

**extracting region proposals using the 'selectivesearch'** 

**Intersection over Union(IoU)**



# Introducing object detection

### Training a typical object detection model involves the following steps

1. Creating ground truth data that contains labels of the bounding box and class corresponding to various objects present in the image.
2. Coming up with mechanisms that can through the image to identify regions (region proposals) that are likely to contain objects. 
3. Creating the target class variable by using the IoU metric.
4. Creating the target bounding box offset variable to make corrections to the location of region proposal coming in the second step.
5. Building a model that can predict the class of object along with the target bounding box offset corresponding to the region proposal.
6. Measuring the accuracy of object detection using **mean Average Precision**

# Creating a bounding box ground truth for training

> Note that when we detect the bounding box, we are detecting the pixel locations of the four corners of the bounding box surrounding the image.

'ybat' to create (annotate) bounding boxes around objects in the image. 



# Understanding region proposals

**Region proposal** is a technique that helps in identifying islands of regions where the pixels are similar to one another.

### how region proposals assist in object localization and detection !?



# Leveraging SelectiveSearch to generate region proposals

SelectiveSearch is a region proposal algorithm used for object localization where it generates proposals of regions that are likely to be grouped together based on their pixel intensities. 

SelectiveSearch over-segments an image by grouping pixels based on the preceding attributes. Next, it iterates through these over-segmented groups and groups them based on similarity. At each iteration, it combines smaller regions to form a larger region.



**'felzenszwalb' segments (which are obtained based on the color, texture, size, and shape compatibility of content within an image) from the image.**

; scale represents the number of clusters that can be formed within the segments of the image. The higher the value of 'scale', the greater the detail of the original image that is preserved.

> Pixels that have similar values form a region proposal. This now helps in object detection, as we now pass each region proposal to a network and ask it to predict whether the region proposal is a background or an object. Furthermore, if it is an object, it would help us to identify the offset to fetch the tight bounding box corresponding to the object and also the class corresponding to the content within the region proposal.



## Implementing SelectiveSearch to generate region proposals

Define 'extract_candidates' function that fetches the region proposals from an image

- Fetch only those candidates that are over 5% of the total image area and less than or equal to 100% of the image area and return them.

```python
for i in regions:
	if r['rect'] in candidates: continue
	if r['size'] < (0.05*img_area): continue
	if r['size'] > (1*img_area): continue
	x, y, w, h = r['rect']
	candidates.append(list(r['rect']))
return candidates
```



![image-20210408143047081](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408143047081.png)



The grids in the preceding diagram represent the candidate regions coming from the 'selective_search' method.

**How do we leverage region proposals for object detection and localization?**



**handy ; 손쉬운 편리한**

# Understanding IoU



![Illustration of intersection-over-union (IOU). | Download Scientific Diagram](https://www.researchgate.net/publication/346512249/figure/fig5/AS:963793576292352@1606797703167/Illustration-of-intersection-over-union-IOU.png)

IoU as a metric is the ratio of the overlapping region over the combined region between the two bounding boxes.

Define 'get_iou' function that takes boxA and boxB as inputs where boxA and boxB are two different bounding boxes ( boxA: ground truth bounding box boxB; region proposal)

'epsilon' parameter to address the rare scenario when the union between the two boxes is 0, resulting in a division by zero error. 

# Non-max suppression

**Non-max** refers to the boxes that do not contain the highest probability of containing an object, and **suppression** refers to us discarding those boxes that do not contain the highest probabilities of containing an object. In non-max suppression, we identify the bounding box that has the highest probability and discard all the other bounding boxes that have an IoU greater than a certain threshold with the box containing the highest probability of containing an object.

In PyTorch, nonmax suppression is performed using the 'mns' function in the 'torchvision.ops' module. The 'nms' function takes the bounding box coordinates, the confidence of the object in the bounding box, and the threshold of IoU across bounding boxes, to identify the bounding boxes to be retained.

# Mean average precision

mAP comes to the rescue in such a scenario. 

- Precision ; 
  $$
  Precision = \frac{True\ positives}{(True\ positives + False\ positives)}
  $$
  A true positive refers to the bounding boxes that predicted the correct class of objects and that have an IoU with the ground truth that is greater than a certain threshold.

  A false positive refers to the bounding boxes that predicted the class incorrectly or have an overlap that is less than the defined threshold with the ground truth.

Furthermore, if there are multiple bounding boxes that are identified for the same ground truth bounding box, only one box can get into a true positive, and everything else gets into a false positive.

- **mAP** ; mAP is the average of precision values calculated at various IoU threshold values across all the classes of objects present within the dataset.



## Training R-CNN-based custom object detectors

R-CNN stands for **Region-based Convolutional Neural Network**. **Region-based** within R-CNN stands for the region proposals. Region proposals are used to identify objects within an image. Note that R-CNN assists in identifying both the objects present in the image and the location of objects within the image.



## R-CNN 

![image-20210408195431756](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408195431756.png)

1. Extract region proposals from an image;
   - Ensure that we extract a high number of proposals to not miss out on any potential object within the image.
2. Resize (warp) all the extracted regions to get images of the same size.
3. Pass the resized region proposals through a network;
   - Typically, we pass the resized region proposals through a pretrained model such as VGG16 or ResNet50 and extract the features in a fully connected layer.
4. Create data for model training, where the input is features extracted by passing the region proposals through a pretrained model, and the outputs are the class corresponding to each region proposal and the offset of the region proposal from the ground truth corresponding to the image;
   - If a region proposal has an IoU greater than a certain threshold with the object, we prepare training data in such a way that the region is responsible for predicting the class of object it is overlapping with and also the offset of region proposal with the ground truth bounding box that contains the object of interest.
5. Connect two output heads, one corresponding to the class of image and the other corresponding to the offset of region proposal with the ground truth bounding box to extract the fine bounding box on the object;
   - 
6. Train the model post, writing a custom loss function that minimizes both object classification error and the bounding box offset error.



> Note that the loss function that we will minimize differs from the loss function that is optimized in the original paper.



# Implementing R-CNN for object detection on a custom dataset

1. Downloading the dataset
2. Preparing the dataset
3. Defining the region proposals extraction and IoU calculation functions
4. Creating the training data
   - Creating input data for the model
     - Resizing the region proposals
     - Passing them through a pretrained model to fetch the fully connected layer values
   - Creating output data for the model
     - Labeling each region proposal with a class or background label
     - Defining the offset of the region proposal from the ground truth if the region proposal corresponds to an object and not background
5. Defining and training the model
6. Predicting on new images

# Fetching region proposals and the ground truth of offset

The input constitutes the candidates that are extracted using the 'selectivesearch' method and the output constitutes the class corresponding to candidates and the offset of the candidate with respect to the bounding box it overlaps the most with if the candidate contains an object. 

FPATHS ;  file paths.

GTBBS ; ground truth bounding boxes.

CLSS ; classes of objects.

DELTAS ; the delta offset of a bounding box with region proposals.

ROIS ; region proposal locations.

IOUS ; IoU of region proposals with ground truths.

- For this exercise, we can use all the data points for training or illustrate with **just the first 500 data points**. You can choose between either of the two, which dictates the training time and training accuracy ( the greater the data points, the greater the training time and accuracy) : in the preceding code, we are specifiying that we will work on 500 images.

- Fetch the offsets needed (delta) to transform the current proposal into the candidate that is the best region proposal(which is the ground truth bounding box) - best_bb, in other words, how much should the left, right, top, and bottom margins of the current proposal be adjusted so that it aligns exactly with best_bb from the ground truth;

  ```python
  delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W, H, W, H])
  deltas.append(delta)
  rois.append(candidate / np.array([W,H,W,H]))
  ```

# Creating the training data

In this section, we will prepare a dataset class based on the ground truth of region proposals that are obtained by the end of step 8 and create data loaders from it. Next, we will normalize each region proposal by resizing them to the same shape and scaling them. we will continue coding from where we left off in the preceding section.



# R-CNN network architecture

1. Define a VGG backbone.

2. Fetch the features post passing the normalized crop through a pretrained model.

3. Attach a linear layer with sigmoid activation to the VGG backbone to predict the class corresponding to the region proposal.

4. Attach an additional linear layer to predict the four bounding box offsets.

5. Define the loss calculations for each of the two outputs(one to predict class and the other to predict the four bounding box offsets)

   ```python
   self.cel = nn.CrossEntropyLoss()
   self.sl1 = nn.L1Loss()
   ```

   

6. Train the model that predicts both the class of region proposal and the four bounding box offsets.



- Define the feed-forward method where we pass the image through a VGG backbone to fetch features , which are further passed through the methods corresponding to classification and bounding box regression to fetch the probabilities across classes and the bounding box offsets;

```python
def forward(self, input):
    feat = self.backbone(input)
    cls_score = self.cls_score(feat)
    bbox = self.bbox(feat)
    return cls_score, bbox
```

# Predict on a new image

1. Extract region proposals from the new image.
2. Resize and normalize each crop.
3. Feed-forward the processed crops to make predictions of class and the offsets.
4. Perform non-max suppression to fetch only those boxes that have the highest confidence of containing an object.



# Training Fast R-CNN-based custom object detectors

![image-20210408214222403](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408214222403.png)

One of the major drawbacks of R-CNN is that it takes considerable time to generate predictions, as generating region proposals for each image, resizing the crops of regions, and extracting features corresponding to **each crop**(region proposal), constitute the bottleneck.

Fast R-CNN gets around this problem by passing the **entire image** through the pretrained model to extract features and then fetching the region of features that correspond to the region proposals of the original image.

# Working details of Fast R-CNN

![The architecture of Faster R-CNN. | Download Scientific Diagram](https://www.researchgate.net/profile/Zhipeng-Deng-2/publication/324903264/figure/fig2/AS:640145124499471@1529633899620/The-architecture-of-Faster-R-CNN.png)

1. Pass the image through a pretrained model to extract features prior to the flattening layer; let's call the output as feature maps
2. Extract region proposals corresponding to the image.
3. Extract the feature map area corresponding to the region proposals (note that when an image is passed through a VGG16 architecture, the image is downscaled by 32 at output as there are 5 pooling operations performed. Thus, if a region exists with a bounding box of (40,32,200,24) in the original images, the feature map corresponding to the bounding box of (5,4,25,30) would correspond to the exact same region).
4. Pass the feature maps corresponding to region proposals through the RoI pooling layer one at a time so that all feature maps of region proposals have a similar shape.This is a replacement for the warping that was executed in the R-CNN technique.
5. Pass the RoI pooling layer output value through a fully connected layer.
6. Train the model to predict the class and offsets corresponding to each region proposal.

> **Note that the big difference between R-CNN and Fast R-CNN is that, in R-CNN, we are passing the crops (resized region proposals) through the pretrained model one at a time, while in Fast R-CNN, we are cropping the feature map (which is obtained by passing the whole image through a pretrained model) corresponding to each region proposal and thereby avoiding the need to pass each resized region proposal through the pretrained model.**



R-CNN, Fast R-CNN are still very slow to be used in real time. This is primarily because we are still using two different models, one to generate region proposals and another to make predictions of class and corrections. 

# Questions

1. How does a region proposal technique generate proposals?
2. How is IoU calculated if there are multiple objects in an image?
3. Why does R-CNN take a long time to generate predictions?
4. Why is Fast R-CNN faster when compared with R-CNN?
5. How does RoI pooling work?
6. What is the impact of not having multiple layers post the feature map obtained when predicting the bounding box coreections?
7. Why do we have to assign a higher weight to regression loss when calculating overall loss?
8. How does non-max suppression work?





