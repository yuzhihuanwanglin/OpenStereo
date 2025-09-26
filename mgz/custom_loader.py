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


def custom_dataloader(root,num_samples):
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    dataset = KITTIStereoDataset(root=root, transform=transform)

    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True)   # 避免多进程拷贝问题
    return dataloader


if __name__ == "__main__":
    kitti_root = "/home/lc/gaoshan/Workspace/dataset/kitti/dataset2015/testing"

    dataloader = custom_dataloader(
        root=kitti_root,
        num_samples=10
    )

    save_dir = "./npy"
    os.makedirs(save_dir, exist_ok=True)

    # 构建 dataloader，每次返回单张左右图像
    loader = custom_dataloader(kitti_root, num_samples=10)

    # 测试迭代
    for i, (left, right) in enumerate(loader):
        print("Left:", left.shape, "Right:", right.shape)  # 输出 (256, 512, 3)
        # 可以直接喂给模型
        left_npy = left.numpy()
        right_npy = right.numpy()

        np.save(os.path.join(save_dir, f"left_{i:03d}.npy"), left_npy)
        np.save(os.path.join(save_dir, f"right_{i:03d}.npy"), right_npy)

        print(f"保存完成：left_{i:03d}.npy, right_{i:03d}.npy")
