import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
import os, glob
import numpy as np
import cv2

class KITTIStereoDataset(Dataset):
    def __init__(self, root, transform=None, resize=(512, 256)):
        self.left_images = sorted(glob.glob(os.path.join(root, "Image0", "*.jpg")))
        self.right_images = sorted(glob.glob(os.path.join(root, "Image1", "*.jpg")))
        assert len(self.left_images) == len(self.right_images), "左右图像数量不匹配"
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        
        left_img = Image.open(self.left_images[idx]).convert("RGB")
        right_img = Image.open(self.right_images[idx]).convert("RGB")
        

       
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            print(left_img.shape)
            print(right_img.shape)    
            return left_img, right_img


def custom_dataloader(root,num_samples):
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    dataset = KITTIStereoDataset(root=root, transform=transform)

    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)   # 避免多进程拷贝问题
    return dataloader


if __name__ == "__main__":
    #kitti_root = "/home/lc/gaoshan/Workspace/dataset/kitti/dataset2015/testing"
    kitti_root = "/home/lc/share/datas/00000"


    save_dir = "./npy"
    os.makedirs(save_dir, exist_ok=True)

    # 构建 dataloader，每次返回单张左右图像
    loader = custom_dataloader(kitti_root, num_samples=5)

    # 测试迭代
    for i, (left, right) in enumerate(loader):
        print("Left:", left.shape, "Right:", right.shape)  # 输出 (256, 512, 3)
        # 可以直接喂给模型
        left_npy = left.numpy()
        right_npy = right.numpy()
        
        print(left_npy)
        #left_npy = np.expand_dims(left_npy, axis=0)
        #right_npy = np.expand_dims(right_npy, axis=0) 
        print(left_npy.shape)
        print(right_npy.shape)

        np.save(os.path.join(save_dir, f"left_{i:03d}.npy"), left_npy)
        np.save(os.path.join(save_dir, f"right_{i:03d}.npy"), right_npy)

        print(f"保存完成：left_{i:03d}.npy, right_{i:03d}.npy")
