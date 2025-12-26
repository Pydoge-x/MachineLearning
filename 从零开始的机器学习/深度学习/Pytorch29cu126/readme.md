# 深度学习配置-Pytorch版

## 确保环境准备

1. 确保安装requirements.txt中的所有依赖包：

   ```bash
   pip install -r requirements.txt
   ```
2. 检查自己电脑是否有GPU，如果有，确保安装了合适版本的CUDA和cuDNN。

3. 安装PyTorch，建议参考[PyTorch官网](https://pytorch.org/get-started/locally/)根据自己的系统和CUDA版本选择合适的安装命令。
4. 安装其他深度学习框架（如TensorFlow、Keras等），根据需要选择安装，本项目基于PyTorch，仅需安装PyTorch即可。
5. 检查PyTorch是否正确安装并能使用GPU：

   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```
   
6. 确保安装Jupyter Notebook以便运行和测试代码：

   ```bash
   pip install notebook
   ```
