from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python usage --> data type: tensor
# transforms.

img_path = "../dataset/train/ants/0013035.jpg"
img_path_abs = "/dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("../logs")

# 1. how to use transforms?
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

# 2. why we need data type tensor?(what is the different between tensor and others data type?)
writer.add_image("Tensor_img", tensor_img)
writer.close()