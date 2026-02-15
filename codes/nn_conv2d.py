import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 初始化数据集，使用dataloader读取
dataset = torchvision.datasets.CIFAR10(root='./pytorch_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Witt(nn.Module):
    def __init__(self):
        # 初始化Witt类，设定Conv2d参数
        super(Witt, self).__init__()
        # 输入通道为3，输出通道为6，卷积核为3x3大小，stride为1，padding=0（不填充）
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

# 实例化Witt类
witt = Witt()
# print(witt)

step = 0
writer = SummaryWriter("../logs/nn_conv2d_logs")

for data in dataloader:
    imgs, targets = data
    output = witt(imgs)
    # print(output.shape)
    # print(imgs.shape)
    writer.add_images("input", imgs, global_step=step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()