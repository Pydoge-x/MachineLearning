import torch
from torch import nn
input = torch.tensor([[1.1, -0.5],
                    [-1, 3 ]])
class Relu(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Relu, self).__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=False) # inplace表示是否修改原始数据

    def forward(self, input):
        output = self.relu1(input)
        return output

model = Relu()
output = model(input)
print(output)