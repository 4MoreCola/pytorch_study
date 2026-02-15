import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]],
                      dtype=torch.float32
                    )

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../pytorch_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Witt(nn.Module):
    def __init__(self):
        super(Witt, self).__init__()
        self.maxpool_c = MaxPool2d(kernel_size=3, ceil_mode=True)
        self.maxpool_f = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output_c = self.maxpool_c(input)
        output_f = self.maxpool_f(input)
        return output_c, output_f

witt = Witt()
output_c, output_f = witt(input)
print(output_c)
print(output_f)

writer = SummaryWriter("../logs/maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output_c, output_f = witt(imgs)
    writer.add_images("output_c", output_c, step)
    writer.add_images("output_f", output_f, step)
    step += 1

writer.close()
