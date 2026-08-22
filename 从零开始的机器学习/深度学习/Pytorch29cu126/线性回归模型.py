import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()

        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, output_dim, requires_grad=True))

    def forward(self, inputs):

        z = torch.matmul(inputs, self.W) + self.bias

        output = torch.sigmoid(z)

        return output