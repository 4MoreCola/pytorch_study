import torch
from torch import nn

# 定义nn.module类的子类
class Witt(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义初始化处理

    def forward(self, input):
        # 定义forward处理
        output = input + 1
        return output

#
witt = Witt()
x = torch.tensor(1.0)
output = witt(x)
print(output)