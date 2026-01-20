from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10(root="./pytorch_dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张 image and target
img, target = test_dataset[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader_logs")

for epoch in range(2):
    step = 0

    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch: {{{epoch}}}", img_tensor=imgs, global_step=step)
        step += 1

writer.close()