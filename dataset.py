
import os, random, torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageRestorationDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, patch_size=256, is_train=True):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.patch_size = patch_size
        self.is_train = is_train
        self.blur_images = sorted(os.listdir(blur_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))
        assert len(self.blur_images) == len(self.sharp_images)
        print(f"Loaded {len(self.blur_images)} pairs from {blur_dir}")

    def __len__(self):
        return len(self.blur_images)

    def augment(self, blur, sharp):
        if random.random() > 0.5:
            blur = TF.hflip(blur); sharp = TF.hflip(sharp)
        if random.random() > 0.5:
            blur = TF.vflip(blur); sharp = TF.vflip(sharp)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            blur = TF.rotate(blur, angle); sharp = TF.rotate(sharp, angle)
        return blur, sharp

    def __getitem__(self, idx):
        blur = Image.open(os.path.join(self.blur_dir, self.blur_images[idx])).convert("RGB")
        sharp = Image.open(os.path.join(self.sharp_dir, self.sharp_images[idx])).convert("RGB")
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(blur, output_size=(self.patch_size, self.patch_size))
            blur = TF.crop(blur, i, j, h, w); sharp = TF.crop(sharp, i, j, h, w)
            blur, sharp = self.augment(blur, sharp)
        else:
            blur = transforms.CenterCrop(self.patch_size)(blur)
            sharp = transforms.CenterCrop(self.patch_size)(sharp)
        to_tensor = transforms.ToTensor()
        return to_tensor(blur), to_tensor(sharp)
