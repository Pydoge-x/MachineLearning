from torch import nn
import torch
# 搭建卷积神经网络
class CIFARmodel(nn.Module):
    def __init__(self):
        super(CIFARmodel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2 ),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == '__main__':
    model = CIFARmodel()
    input = torch.ones(64, 3, 32, 32)
    output = model(input)
    print(output.shape)
