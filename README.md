# TinyImageNet-Pytorch-VGGNet

TinyImageNet Dataset을 이용하여 pytorch로 VGGNet을 만들고 학습합니다.

## TinyImageNet

TinyImageNet은 기존의 224x224 이미지가 총 1300개씩 1000개의 클래스가 있는 ImageNet이 너무 용량이 크고 학습하기 어려워서 나온 작은 사이즈의 Dataset입니다.

TinyImageNet은 64x64 이미지가 총 1000개씩 100개의 클래스가 있습니다. ImageNet보다 적은 용량입니다. (ImageNet은 train set만 130GB이고 TinyImageNet은 총 합쳐도 400MB정도입니다.)

TinyImageNet은 https://tiny-imagenet.herokuapp.com/ 여기서 받을 수 있지만, train 데이터와 validation 데이터가 0~100 클래스 폴더로 나뉘어 있는 https://www.kaggle.com/c/thu-deep-learning/data 여기의 데이터를 사용하였다.

## VGGNet

VGGNet은 2014년도에 개최된 ILSVRC'14 에서 준우승을 한 총 19개의 Layer로 이루어진 CNN 구조입니다.

VGGNet의 기본 원리는 CNN에서 깊이가 깊을수록 더 많은 정보를 저장할 수 있다는 것입니다.

그래서 AlexNet보다 작은 컨볼루션 필터를 사용하여서 깊이를 늘렸습니다. 이는 7x7 필터를 사용하는 것과 3x3 필터를 여러번 사용하는 것이 더 많은 깊이를 만들 수 있다는 것에서 시작되었습니다.

32x32 이미지에서 7x7 필터를 사용할 경우 한번의 컨볼루션으로 32x32 -> 26x26 으로 이미지는 줄어들지만,

3x3필터를 사용할 경우 32x32 -> 26x26으로 줄이기 위해서는 총 3번의 컨볼루션이 필요합니다.

즉, 깊이가 늘어나는 것입니다.

이를 통해서 VGGNet은 다음과 같은 구조를 채택하였습니다.

![img1](https://github.com/kjo26619/TinyImageNet-Pytorch-VGGNet/blob/main/image/vgg.png)

VGGNet에서는 깊이의 차이를 위해 다양한 Layer 수를 사용하였습니다.

conv3-64의 의미는 3x3x64 Convlution Filter를 통해 이미지를 Convolution 했다는 것입니다.

VGGNet은 간단한 구조이지만 높은 효율을 보입니다.

# TinyImageNet Pytorch



