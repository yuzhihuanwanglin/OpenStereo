import cv2
import numpy as np
from PIL import Image

# 输入参数
input_size = (512, 256)  # (W,H)
normalize_mean = [0.485, 0.456, 0.406]
normalize_std  = [0.229, 0.224, 0.225]

# NPU int8 输入量化参数
scale_left  = 0.017425
scale_right = 0.017288
zero_point  = 0

def preprocess_to_int8(img_path, scale, zero_point, size=(512,256)):
    """
    将输入RGB图像处理为符合NPU要求的 int8 tensor (1,3,256,512)
    """
    # 1. 读取并 resize
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)

    # 2. 转为 numpy float32，范围 [0,1]
    img = np.array(img).astype(np.float32) / 255.0

    # 3. Normalize (减均值 / 除标准差)
    img[..., 0] = (img[..., 0] - normalize_mean[0]) / normalize_std[0]
    img[..., 1] = (img[..., 1] - normalize_mean[1]) / normalize_std[1]
    img[..., 2] = (img[..., 2] - normalize_mean[2]) / normalize_std[2]

    # 4. HWC -> CHW
    img = img.transpose(2,0,1)   # (3,H,W)

    print(f"-------------------")
    print(img)

    # 5. 量化 float32 -> int8
    img_int8 = np.round(img / scale + zero_point).astype(np.int8)

    # 6. 加 batch 维度
    img_int8 = np.expand_dims(img_int8, axis=0)  # (1,3,H,W)

    return img_int8

# 示例用法
left_path  = "/home/lc/share/datas/00000/Image0/00050.jpg"
right_path = "/home/lc/share/datas/00000/Image1/00050.jpg"

left_int8  = preprocess_to_int8(left_path,  scale_left,  zero_point)
right_int8 = preprocess_to_int8(right_path, scale_right, zero_point)

print("Left input shape :", left_int8.shape, left_int8.dtype)
print("Right input shape:", right_int8.shape, right_int8.dtype)

# 如果需要保存到二进制文件（方便NPU加载）

np.save('left_int8.npy', left_int8)
np.save('right_int8.npy', right_int8)
# left_int8.tofile("left_img_input.bin")
# right_int8.tofile("right_img_input.bin")
