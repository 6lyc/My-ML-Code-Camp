from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, layers):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3_drop = nn.Dropout2d()

        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_drop(x)
        #一个channel或者一个feature map代表一个神经元
        #drop:Randomly zero out entire channels:A channel is a 2D feature map.
        #Each channel will be zeroed out independently on every forward call with probability :attr:`p` using samples from a Bernoulli distribution
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

