import torch
import torch.nn.functional as F

# 构造一个数字图像矩阵
input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])

# 构造卷积核
kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

input = torch.reshape(input, (1, 1, 5, 5)) # shape:(batch_size=1, channels=1, elements_in_array=5, arrays_num=5)
kernel = torch.reshape(kernel, (1, 1, 3, 3)) # shape:(batch_size=1, channels=1, elements_in_array=3, arrays_num=3)
print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print(f"stride=1, {output1}")


output2 = F.conv2d(input, kernel, stride=2)
print(f"stride=2, {output2}")

output_padding = F.conv2d(input, kernel, stride=1, padding=1)
print(f"padding=1, {output_padding}")