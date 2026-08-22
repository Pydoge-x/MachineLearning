import torch 
from torch import nn as nn 

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

mynn = MyNN()
output = mynn(input)
print(output)
    