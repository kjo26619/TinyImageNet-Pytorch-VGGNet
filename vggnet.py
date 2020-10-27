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
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

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
