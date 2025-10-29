import numpy as np
from PIL import Image
from torchvision import transforms
import os

# ========== 配置 ==========
in_left = "/home/fays007/lc/share/mgz/datas/left.png"
in_right = "/home/fays007/lc/share/mgz/datas/right.png"
out_dir = "output/quant/"
os.makedirs(out_dir, exist_ok=True)


input_size = (512, 256)  # (W,H)

# input_size = (256, 160)  # (W,H)
# 手动指定 scale
scale_left = 0.000081
scale_right = 0.000081

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# ========== 预处理函数 ==========
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((input_size[1],input_size[0])),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    return np.expand_dims(t(img).numpy(), axis=0)  # shape (1,3,H,W)

# ========== 量化函数 ==========
def quantize_int16(arr, scale):
    arr_q = np.clip(np.round(arr / scale), -32768, 32767).astype(np.int16)
    return arr_q

# ========== 主流程 ==========
left_f32 = preprocess(in_left)
right_f32 = preprocess(in_right)

left_i16 = quantize_int16(left_f32, scale_left)
right_i16 = quantize_int16(right_f32, scale_right)

# 保存结果
np.save(out_dir + "left_32.npy", left_f32)
np.save(out_dir + "right_32.npy", right_f32)

left_i16.tofile(out_dir + f"left_16.bin")
right_i16.tofile(out_dir + f"right_16.bin")

np.save(out_dir + f"left_16.npy", left_i16.astype(np.float32))
np.save(out_dir + f"right_16.npy", right_i16.astype(np.float32))

print("✅ int16 量化完成")
print(f"Left scale = {scale_left}, Right scale = {scale_right}")
print(f"Left shape = {left_i16.shape}, Right shape = {right_i16.shape}")
