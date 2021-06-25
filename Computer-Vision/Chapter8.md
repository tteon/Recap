# Chapter8



## Components of modern object detection algorithms



R-CNN  , Fast R-CNN techniques is that they have two disjointed networks - 하나는 오브젝트가 포함된 지역을 판별하는 것이고 나머지는 오브젝트가 나타난 곳 즉 바운딩 박스를 교정하는 것이다. 더욱이 두 모델은 region proposal과 같은 많은 forward propagation이 필요하다. 현대 object detection 알고리즘은 주로 single neural network 훈련하는 것에 강한 의존도를 가지고 있고 또한 단 한번의 forward pass를 통해 모든 object 를 detect하는 능력을 가지고 있음.

본 section 에서 배울것들

- Anchor boxes
- Region proposal network (RPN)
- Region of interest pooling

### Anchor boxes

'selectivesearch' Anchor boxes come in as a handy replacement for selective search -> 'selectivesearch' 기반 영역제안을 어떻게 대체할것인가 이번 섹션에서 배우도록 한다.

이미지들은 대체적으로 비슷한 shape를 갖는데 이 가정을 시작으로 training 전에 width , height -> object를 유추해내고자 함. 

우리가 object의 height , width aspect ratio 에 대한 아이디어를 토대로 우리의 custom image를 나타낸다, 우리는 anchor boxes 를 height & width 로 dataset 내에 있는 bounding box 와 비교하며 유추해낸다 ? 라고 이해를 함.

anchor box obtain process

1. 이미지의 top left 부터 bottom right까지 slide를 통해 anchor box를 생성함.
2. high intersection over union ( IoU ) with the object 는 label을 갖는다 다른 IoU 는 0 해당하는 object 는 1 ?!
   - IoU 가 특정 threshold 보다 크면 object class 는 1 , 낮으면 0 .인데 이때 threshold 를 직접 handling 가능함. 



varying scales the anchor box 를 통해 이미지 내의 object 를 accommodate 할 수 있음.



### Region Proposal Network

224 x 224 x 3 -> anchor box 의 shape 8 x 8 은 이를 위한 것임. 만약 우리가 8 픽셀의 stride를 가지면 우리는 224/8 = 28 , 즉 28개의 crops를 매 row 마다 가져올 수 있음. 정리하자면 28 * 28 을 통해 576 개의 crop을 picture로 부터 가져올 수 있게 된다. 우리는 그때 그 crops들을 가져오고 Region Proposal Network model(RPN)을 활용하여 그 crop이 image에 포함되어있는지 안되어있는지를 분별한다. 특별히 RPN은 object가 가진 crop의 likelihood를 추정하게 됨.



selectivesearch와 output of an RPN을 비교해보자면 다음과 같음.

'selectivesearch'는 top of pixel value에 기반하여 region candidate를 제공한다. 허나 RPN은 anchor boxes 그리고 이미지 내의 slide 를 활용하여 적용한 다양한 anchor box를 활용한 후 region candidate 를 제공함. 우리가 앞선 2가지 방법을 통해 region candidate에 대해 얻을때, 그 candidate가 어떤 object인지 분별이 가능함. 앞 선 프로세스를 활용하여!



Region proposal generation 에 기반한 'selectivesearch' 는 neural network 에서 벗어난 방법론임, 우리는 object detection network 의 부분 중 하나인 RPN을 build할 수 있다. RPN을 활용해서, 우리는 이제 불필요한 연산을 할 필요가 없어졌다 region proposal을 계산하지않아도되는 . 이 방법은 regions identify , identify classes of objects in image , identify their corresponding bounding box location 등을 진행할 수 있는 model을 가지게 됨.



RPN identifies 가 어떻게 region candidate를 통해 특정 crop 내에 object가 포함 되어있는지 안되어있는지에 대해 배우는것에 대해 배우게 될것임. training data에서는 gt(ground truth)를 object와 일치시키고 그 다음은 candidate와 ground truch 의 bounding boxes 를 iou metric 통해 evaluation 함. 이때 threshold 이상이면 1 아니면 0. 

region candidate 를 통해 object에 대해 predict할 때 우리는 non-max suppression 을 활용하여 object 내의 overlapping을 포함할수있다..? 

Summary

1. 다른 관점 비율과 사이즈를 특정 이미지로부터 crop한 것을 토대로 anchor box를 sliding 해줌.
2. 이전 anchor box를 통해 생성된 crop과 ground truth bounding box의 object를 IoU metric 과정을 통해 값을 도출해냄.
3. IoU 의 특정 threshold 이상일 시 object , 아닐시 다른 class object로 간주함.
4. region 내의 object가 포함되어있을지에 대해 모델을 훈련시킴
5. non-max suppression 을 활용하여 region candidate 를 identify함 그리고 다른 region candidate 와 high overlap 한 것을 제거함.(중복제거)



## Classification and regression



앞선 과정에서도 여러 이슈가 발생하나 대략 2가지를 가져와보자면,

1. region proposal은 tightly 하게 object와 일치하지 않을 수 있음. (IoU > 0.5 같은 대략적인 수치를 통해 apporximation 하기때문에.)
2. 우리는 region에 object가 포함되어 있는지 안되어있는지 확인한다, 허나 object의 클래스가 region 안에 포함안되어있을수도 있음.



위 2가지 이슈를 해결해보고자 우리는 network를 통과시켜 uniformly 하게 feature map을 obtain 함. 그 network가 region 내의 object의 class를 예측할것인지에 대해 기대하게 됨. 그리고 offset 또한 region , bounding box와 일치시키도록 최대한 tightly 하게 build해줌.

2개의 task

1. Class of object in the region
2. Amount of offset to be done on the predicted bounding boxes of the region to maximize the IoU with the ground truth.

예를 들자면 20개의 클래스가 데이터의 답이라면 그 neural network는 총 25개의 아웃풋을 도출해냄 - 21개는 특정 클래스 그리고 4개는 bounding box의 height , width 그리고 2개의 center 좌표 



## Training Faster R-CNN on a custom dataset



```python
label2target = {l:t+1for t, l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t : lfor i,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)
```



































### 



### 

