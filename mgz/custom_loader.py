import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
import os, glob
import numpy as np
import cv2

class KITTIStereoDataset(Dataset):
    def __init__(self, root, transform=None, resize=(512, 256)):
        self.left_images = sorted(glob.glob(os.path.join(root, "image_2", "*.png")))
        self.right_images = sorted(glob.glob(os.path.join(root, "image_3", "*.png")))
        assert len(self.left_images) == len(self.right_images), "左右图像数量不匹配"
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_images[idx]).convert("RGB")
        right_img = Image.open(self.right_images[idx]).convert("RGB")
        
        
        left_img = np.array(left_img, dtype=np.float32)
        right_img = np.array(right_img, dtype=np.float32)

        if self.resize:
            left_img = cv2.resize(left_img,self.resize, Image.BILINEAR)
            right_img = cv2.resize(right_img,self.resize, Image.BILINEAR)


        # print(left_img.shape)
        # print(right_img.shape)
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img


def custom_dataloader(normalize_mean, normalize_std, batch_size,
                      num_samples, num_workers, kitti_root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    dataset = KITTIStereoDataset(root=kitti_root, transform=transform)

    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)   # 避免多进程拷贝问题
    return dataloader


if __name__ == "__main__":
    kitti_root = "/media/wanglin/Elements/datasets/KITTI2015/testing"

    dataloader = custom_dataloader(
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        batch_size=1,
        num_samples=10,
        num_workers=1,   
        kitti_root=kitti_root
    )

    for left, right in dataloader:
        print(left.shape, right.shape)  # torch.Size([B, 3, H, W])
