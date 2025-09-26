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


left_img = Image.open('/home/extra/share/datas/left.png').convert("RGB")
right_img = Image.open('/home/extra/share/datas/right.png').convert("RGB")


#### 图片转RGB uint8
left_img_array = np.array(left_img.resize((512, 256)))  # 注意：PIL的resize参数是 (宽, 高)
left_img_uint8 = torch.from_numpy(left_img_array).permute(2, 0, 1)  # 转为 CHW 格式

right_img_array = np.array(right_img.resize((512, 256)))  # 注意：PIL的resize参数是 (宽, 高)
right_img_uint8 = torch.from_numpy(right_img_array).permute(2, 0, 1)  # 转为 CHW 格式

left_uint8_npy = np.expand_dims(left_img_uint8, axis=0)
right_uint8_npy = np.expand_dims(right_img_uint8, axis=0) 


print(left_uint8_npy.shape)
print(right_uint8_npy.shape)

print(left_uint8_npy)
print(right_uint8_npy)

left_uint8_npy.tofile("output/quant/left_uint8.npy")
right_uint8_npy.tofile("output/quant/right_uint8.npy")




# left_int8.tofile("output/quant/left.npy")

#### 图片转RGB 量化int8
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

left_npy_4 = np.expand_dims(left_npy, axis=0)
right_npy_4 = np.expand_dims(right_npy, axis=0) 


np.save('output/quant/left.npy', left_npy_4)
np.save('output/quant/right.npy', right_npy_4)

print('numpy数据是:')
print(left_npy_4.shape)
print(right_npy_4.shape)
print(left_npy)
print(right_npy)

#float32 转 int8
scale_left = 0.020706
scale_right = 0.020706

left_npy_2 = left_npy/scale_left
right_npy_2 = right_npy/scale_right

print(left_npy_2.dtype, left_npy_2.min(), left_npy_2.max())
print(right_npy_2.dtype, right_npy_2.min(), right_npy_2.max())



left_int8 = np.clip(np.round(left_npy / scale_left), -128, 127).astype(np.int8)
right_int8 = np.clip(np.round(right_npy / scale_right), -128, 127).astype(np.int8)


left_float = np.clip(np.round(left_npy / scale_left), -128, 127).astype(np.float32)
right_float = np.clip(np.round(right_npy / scale_right), -128, 127).astype(np.float32)


left_num = float(np.dot(left_npy.flatten(),left_float.flatten()))
left_demon = np.linalg.norm(left_npy.flatten())*np.linalg.norm(left_float.flatten())

print(left_num/left_demon)


right_num = float(np.dot(right_npy.flatten(),right_float.flatten()))
right_demon = np.linalg.norm(right_npy.flatten())*np.linalg.norm(right_float.flatten())

print(right_num/right_demon)


print(left_int8.dtype, left_int8.min(), left_int8.max())
print(right_int8.dtype, right_int8.min(), right_int8.max())

print(left_int8)
print(right_int8)

left_int8 = np.expand_dims(left_int8, axis=0)
right_int8 = np.expand_dims(right_int8, axis=0) 

print(left_int8)
print(right_int8)

left_int8.tofile("output/quant/left.bin")
right_int8.tofile("output/quant/right.bin")





# np.save('output/quant/left.bin', left_int8)
# np.save('output/quant/right.bin', right_int8)
print('LIANGHUA数据是:')
print(left_npy.shape)
print(right_npy.shape)

print(left_int8.shape)
print(right_int8.shape)





