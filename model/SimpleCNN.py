import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, output_num=10, input_channels=1):
        super().__init__()
        # 卷积层1：输入28*28*input_channels 卷积输出24*24*32 池化输出12*12*32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 卷积层2：输入12*12*32 卷积输出10*10*64 池化输出 5*5*64=1600
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 1600
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=output_num),
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将卷积层的输出展平，输入分类器中
        feature = x.view(x.size(0), -1)
        x = self.fc1(feature)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feature