import torch
from torch.utils.data import Dataset
from PIL import Image
import os, glob
import numpy as np

class KITTIStereoDataset(Dataset):
    """
    KITTI 双目图像数据集，每次返回单张左右图像 (H, W, C)
    """
    def __init__(self, root, resize=(512, 256)):
        self.left_images = sorted(glob.glob(os.path.join(root, "image_2", "*.png")))
        self.right_images = sorted(glob.glob(os.path.join(root, "image_3", "*.png")))
        assert len(self.left_images) == len(self.right_images), "左右图像数量不匹配"
        self.resize = resize

    def __len__(self):
        print(len(self.left_images))
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_images[idx]).convert("RGB")
        right_img = Image.open(self.right_images[idx]).convert("RGB")

        if self.resize:
            left_img = left_img.resize(self.resize, Image.BILINEAR)
            right_img = right_img.resize(self.resize, Image.BILINEAR)

        # 转成 numpy array，HWC 格式，float32，[0,1]
        left_np = np.array(left_img).astype(np.float32) 
        right_np = np.array(right_img).astype(np.float32) 
        
        print('__getitem__')
        left_np = torch.from_numpy(left_np)
        right_np = torch.from_numpy(right_np)

        return left_np, right_np


class SingleSampleDataLoader:
    """
    包装 Dataset，使每次迭代返回单张图像对 (H, W, C)
    """
    def __init__(self, dataset, num_samples=None):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.indices = torch.randperm(len(dataset))[:self.num_samples]

    def __iter__(self):
        for idx in self.indices:
            left, right = self.dataset[idx]
            yield left, right

    def __len__(self):
        print(self.num_samples)
        return self.num_samples


def custom_dataloader(root, num_samples=10):
    """
    返回 KITTI 双目图像迭代器，每次迭代返回单张图像对 (H, W, C)
    """
    dataset = KITTIStereoDataset(root=root)
    loader = SingleSampleDataLoader(dataset, num_samples=num_samples)
    return loader


if __name__ == "__main__":
    print('main')
    
    # apt-get install -y libglib2.0-0 libglib2.0-dev
    # KITTI 数据集路径（修改为你本地路径）
    # kitti_root = "/media/wanglin/Elements/datasets/KITTI2015/testing"
    kitti_root = "/share/datas/testing"

    # 构建 dataloader，每次返回单张左右图像
    loader = custom_dataloader(kitti_root, num_samples=10)

    # 测试迭代
    for left, right in loader:
        print("Left:", left.shape, "Right:", right.shape)  # 输出 (256, 512, 3)
        # 可以直接喂给模型
