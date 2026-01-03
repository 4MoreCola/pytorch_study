from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path1 = "dataset/train/ants/5650366_e22b7e1065.jpg"
img_path2 = "dataset/train/bees/16838648_415acd9e3f.jpg"
img_path3 = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(img_path1)
img_arr = np.array(img_PIL)
print(type(img_arr))
print(img_arr.shape)

writer.add_image("train", img_arr, 1, dataformats='HWC')

# y = x
#
# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)

writer.close()

