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


left_img = Image.open('/home/lc/share/datas/left.png').convert("RGB")
right_img = Image.open('/home/lc/share/datas/right.png').convert("RGB")


#### 图片转RGB uint8
left_img_array = np.array(left_img.resize((512, 256)))  # 注意：PIL的resize参数是 (宽, 高)
left_img_uint8 = torch.from_numpy(left_img_array).permute(2, 0, 1)  # 转为 CHW 格式

right_img_array = np.array(right_img.resize((512, 256)))  # 注意：PIL的resize参数是 (宽, 高)
right_img_uint8 = torch.from_numpy(right_img_array).permute(2, 0, 1)  # 转为 CHW 格式

left_uint8_npy = np.expand_dims(left_img_uint8, axis=0)
right_uint8_npy = np.expand_dims(right_img_uint8, axis=0) 


# print(left_uint8_npy.shape)
# print(right_uint8_npy.shape)

# print(left_uint8_npy)
# print(right_uint8_npy)

# left_uint8_npy.tofile("output/quant/left_uint8.npy")
# right_uint8_npy.tofile("output/quant/right_uint8.npy")

print(f" -----------------------------------------------------")


normalize_mean=[0.485, 0.456, 0.406]
normalize_std=[0.229, 0.224, 0.225]
transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])        
        
left_img = transform(left_img)
right_img = transform(right_img)


left_npy = left_img.numpy()
right_npy = right_img.numpy()

left_npy_4 = np.expand_dims(left_npy, axis=0)
right_npy_4 = np.expand_dims(right_npy, axis=0) 

#保存图片前处理后的 float32数据
np.save('output/quant/left_32.npy', left_npy_4)
np.save('output/quant/right_32.npy', right_npy_4)



print('图片前处理后的 float32 numpy数据是:')
print(left_npy_4.shape)
print(right_npy_4.shape)
print(left_npy_4)
print(right_npy_4)

print(f" -----------------------------------------------------")
#float32 转 int8
scale_left = 0.017674
zp_left = 120
scale_right = 0.017207
zp_right = 123



left_int8 = np.clip(np.round(left_npy / scale_left + zp_left), 0, 255).astype(np.uint8)
right_int8 = np.clip(np.round(right_npy / scale_right + zp_right), 0, 255).astype(np.uint8)


print(left_int8.dtype, left_int8.min(), left_int8.max())
print(right_int8.dtype, right_int8.min(), right_int8.max())

left_int8 = np.expand_dims(left_int8, axis=0)
right_int8 = np.expand_dims(right_int8, axis=0) 

print(left_int8)
print(right_int8)

left_int8.tofile(f"output/quant/left_8.bin")
right_int8.tofile(f"output/quant/right_8.bin")
# np.save('output/quant/right_8.npy', right_int8)

np.save(f"output/quant/left_8_float.npy", left_int8.astype(np.float32))
np.save(f"output/quant/right_8_float.npy", right_int8.astype(np.float32))

print(f" -----------------------------------------------------")

# 两个数据的余弦计算
# left_float = np.clip(np.round(left_npy_4 / scale_left + zp_left), 0, 255).astype(np.float32)   #  left_int8.astype(np.float32)
# right_float = np.clip(np.round(right_npy_4 / scale_right  + zp_right), 0, 255).astype(np.float32)

# 反量化回 float32
left_float = (left_int8.astype(np.float32) - zp_left) * scale_left
right_float = (right_int8.astype(np.float32) - zp_right) * scale_right

print(left_float)
print(right_float)


print(left_npy_4.shape)
print(left_float.shape)
print(right_npy_4.shape)
print(right_float.shape)

left_num = float(np.dot(left_npy_4.flatten(),left_float.flatten()))
left_demon = np.linalg.norm(left_npy_4.flatten())*np.linalg.norm(left_float.flatten())
print(left_num)
print(left_num/left_demon)


right_num = float(np.dot(right_npy_4.flatten(),right_float.flatten()))
right_demon = np.linalg.norm(right_npy_4.flatten())*np.linalg.norm(right_float.flatten())

print(right_num/right_demon)





