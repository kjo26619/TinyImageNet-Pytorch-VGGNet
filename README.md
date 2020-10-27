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

TinyImageNet을 사용하기 위해 먼저 다운로드를 받은 뒤 폴더의 위치를 편한 곳에 옮겨둡니다.

그 다음 torchvision에서 dataset을 만드는 것을 지원해주는 ImageFolder와 DataLoader를 사용하여 만듭니다.

```
import torchvision
from torch.utils.data import DataLoader

def data_load():
    train_set = torchvision.datasets.ImageFolder(
        root='./TinyImageNet/train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    )

    test_set = torchvision.datasets.ImageFolder(
        root='./TinyImageNet/val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=50, num_workers=8)

    test_loader = DataLoader(test_set, shuffle=True)

    return train_loader, test_loader
```

저는 확실한 val 폴더에 있는 사진들이 이미 class로 나뉘어져있어서 그냥 val을 test/validaiton set으로 사용했습니다.

pytorch에서는 torchvision과 DataLoader를 통해서 쉽게 데이터를 불러와서 사용할 수 있습니다.

# VGGNet Pytorch

```
import torch.nn as nn

class VGG_Net(nn.Module):
    def __init__(self):
        super(VGG_Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)

        )

        self.avg_pool = nn.AvgPool2d(3)

        self.fc1 = nn.Linear(512 * 1 * 1, 1028)

        self.fc2 = nn.Linear(1028, 1028)

        self.fc3 = nn.Linear(1028, 512)
        self.fc_relu = nn.ReLU()
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        features = self.conv(x)
        x = self.avg_pool(features)
        #print(x.shape)
        x = x.view(features.size(0), -1)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.fc2(x)
        x = self.fc_relu(x)
        x = self.fc3(x)
        x = self.fc_relu(x)
        x = self.classifier(x)

        return x, features
```

pytorch에서 지원하는 구조를 사용해도 크게 상관은 없으나 직접 만들어보는 것이 중요하기에 설계해 보았습니다.

pytorch에서는 torch.nn.module을 상속받아서 직접 모델을 구성할 수 있습니다.

클래스의 __init__ 에서 사용할 Layer 구조들을 만들어둡니다.

그리고 def forward(x)를 통해서 Layer를 구성합니다.


