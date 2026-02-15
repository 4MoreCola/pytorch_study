from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.img_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        name, ext = os.path.splitext(img_name)
        img_label_name = name + ".txt"
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)
        img_item_label_path = os.path.join(self.root_dir, self.label_dir, img_label_name)
        img = Image.open(img_item_path)

        with open(img_item_label_path, 'w') as file:
            file.write(self.img_dir)
            file.close()

        return img, img_name, img_item_label_path

    def __len__(self):
        return len(self.img_path)

root_dir = "../dataset/train"
ants_dir = "ants"
bees_dir = "bees"
ants_label_dir = "ants_label"
bees_label_dir = "bees_label"
ants_dataset = MyData(root_dir, ants_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset

for batch in train_dataset:
    img, img_name, img_item_label_path = batch
    print(img_name)
    print(img_item_label_path)
