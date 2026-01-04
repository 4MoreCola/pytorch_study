from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/bruce-tang-nKO_1QyFh9o-unsplash.jpg")
print(img)

# ToTensor
trans_to_tensor = transforms.ToTensor()
tensor_img = trans_to_tensor(img)
writer.add_image("ToTensor", tensor_img)

#Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
img_norm = trans_norm(tensor_img)
writer.add_image("Normalize", img_norm, global_step = 0)

trans_norm = transforms.Normalize((1, 3, 5), (5, 3, 1))
img_norm = trans_norm(tensor_img)
writer.add_image("Normalize", img_norm, global_step = 1)

trans_norm = transforms.Normalize((9, 3, 6), (6, 3, 5))
img_norm = trans_norm(tensor_img)
writer.add_image("Normalize", img_norm, global_step = 2)
#print(img_norm[0][0][0])

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
trans_img_resize = trans_to_tensor(img_resize)
writer.add_image("Resize", trans_img_resize, global_step = 0)
print(trans_img_resize)

# Compose - resize - 2
trans_resize2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize2, trans_to_tensor])
img_resize2 = trans_compose(img)
writer.add_image("Resize", img_resize2, global_step = 1)

# RandomCrop
trans_random = transforms.RandomCrop((1920, 1080))
trans_compose2 = transforms.Compose([trans_random, trans_to_tensor])
for i in range(10):
    img_crop = trans_compose2(img)
    writer.add_image("RandCropHD", img_crop, global_step = i)

writer.close()