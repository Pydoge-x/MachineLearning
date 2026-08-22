import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import time
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



# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../Datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../Datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练数据集的长度为{test_data_size}")
print(f"测试数据集的长度为{test_data_size}")

# 利用Dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model = CIFARmodel()
if torch.cuda.is_available():
    model.cuda()
print(model)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-3
optimzer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0
total_test_step = 0
epoch = 10
# 添加tensorboard
writer = SummaryWriter("../train_logs")

for i in range(epoch):
    start_time = time.time()
    if i == 0:
        print("cuda是否可以使用:", torch.cuda.is_available())

    total_loss = 0.0
    print(f"=====第{i + 1}轮训练开始=====")
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数为{total_train_step}时，Loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    total_accuracy = 0

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    end_time =time.time()
    print(f"整体测试集上的Loss:{total_loss}")
    print(f"整体测试集上的Accuracy:{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model.state_dict(), "model_{}.pth".format(i))
    print("模型已保存")
    print(f"第{i+1}轮训练使用时间为:{end_time-start_time}")

writer.close()

