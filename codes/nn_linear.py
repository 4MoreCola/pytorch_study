import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
import torch
from torch import nn

dataset = torchvision.datasets.CIFAR10(root='../pytorch_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class LinearPractise(nn.Module):
    def __init__(self):
        super(LinearPractise, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

linear_pra = LinearPractise()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    ouput = linear_pra(output)
    print(output.shape)
