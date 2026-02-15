import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

dataset = torchvision.datasets.CIFAR10("../pytorch_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Witt(nn.Module):
    def __init__(self):
        super(Witt, self).__init__()
        self.relu = ReLU()
        self.sigmid = Sigmoid()

    def forward(self, input):
        # output = self.relu(input)
        output = self.sigmid(input)
        return output

witt = Witt()
# output = witt(input)
# print(output)

writer = SummaryWriter("../logs/no_linear_logs")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = witt(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
