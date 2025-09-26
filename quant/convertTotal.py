import numpy as np
from PIL import Image
from torchvision.transforms.functional import normalize
import torch
import cv2

#  方式1
# 读取图片
# img = Image.open("datas/right.png").convert("RGB")
# img = np.array(img, dtype=np.uint8)   # shape = (H, W, 3)

# # 存为bin
# img.tofile("right.bin")

# 方式 2
# left_img = np.array(Image.open("/home/lc/share/mgz/datas/right.png").convert('RGB'), dtype=np.uint8)

# left_img = cv2.resize(left_img, (512,256),cv2.INTER_LINEAR)
    
    
#     # sample = transform(sample)
# left_img = torch.from_numpy(left_img)
    
    
# left = left_img.permute(2, 0, 1)

# left  = left.squeeze().cpu().numpy()

# print(left)
# left.tofile("right.bin")


# from PIL import Image

# # 打开 PNG
# img = Image.open("/home/lc/share/mgz/datas/right.png")

# # 转成 BMP（不压缩）
# img.save("right.bmp", format="BMP")
from torchvision import transforms

# left_img = Image.open('/home/lc/share/mgz/datas/left.png').convert("RGB")
# right_img = Image.open('/home/lc/share/mgz/datas/right.png').convert("RGB")


left_img = Image.open('/home/lc/share/datas/00000/Image0/00050.jpg').convert("RGB")
right_img = Image.open('/home/lc/share/datas/00000/Image1/00050.jpg').convert("RGB")



normalize_mean=[0.485, 0.456, 0.406]
normalize_std=[0.229, 0.224, 0.225]
transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])        
        
left_img = transform(left_img)
right_img = transform(right_img)



print(left_img)
print(right_img)

left_npy = left_img.numpy()
right_npy = right_img.numpy()

print(left_npy)
print(right_npy)

#float32 转 int8
scale_left = 0.018382
scale_right = 0.018929

left_npy_2 = left_npy/scale_left
right_npy_2 = right_npy/scale_right

print(left_npy_2.dtype, left_npy_2.min(), left_npy_2.max())
print(right_npy_2.dtype, right_npy_2.min(), right_npy_2.max())



left_int8 = np.clip(np.round(left_npy / scale_left), -128, 127).astype(np.int8)
right_int8 = np.clip(np.round(right_npy / scale_right), -128, 127).astype(np.int8)

print(left_int8.dtype, left_int8.min(), left_int8.max())
print(right_int8.dtype, right_int8.min(), right_int8.max())

print(left_int8)
print(right_int8)

left_int8 = np.expand_dims(left_int8, axis=0)
right_int8 = np.expand_dims(right_int8, axis=0) 

print(left_int8)
print(right_int8)

left_int8.tofile("left.bin")
right_int8.tofile("right.bin")

# np.save('~/left.bin', left_int8)
# np.save('~/right.bin', right_int8)

print(left_npy.shape)
print(right_npy.shape)

print(left_int8.shape)
print(right_int8.shape)





