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
scale_left = 0.018524
scale_right = 0.018524
zp_left = 112   # 你可以根据需要调整 zero_point
zp_right = 112  # 通常中点设为128

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# ========== 预处理函数 ==========
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((input_size[1], input_size[0])),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    return np.expand_dims(t(img).numpy().astype(np.float32), axis=0)  # (1,3,H,W)

# ========== 量化函数 (uint8) ==========
def quantize_to_uint8(arr_float, scale, zero_point, verbose=True):
    """
    arr_float: numpy array (float32)
    scale: float 标量
    zero_point: int, 0~255
    """
    if scale == 0:
        raise ValueError("scale must be non-zero")

    arr = arr_float.astype(np.float64)
    q = np.round(arr / float(scale)) + float(zero_point)
    q = np.clip(q, 0, 255)
    q_u8 = q.astype(np.uint8)

    if verbose:
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        q_min, q_max = int(np.min(q_u8)), int(np.max(q_u8))
        z_map = int(np.round(0.0 / float(scale) + float(zero_point)))
        print(f"[quantize_to_uint8] float range: [{a_min:.6f}, {a_max:.6f}]")
        print(f"[quantize_to_uint8] uint8 range after quantize: [{q_min}, {q_max}]")
        print(f"[quantize_to_uint8] scale={scale}, zero_point={zero_point} -> 0.0 maps to {z_map}")

    return q_u8

# ========== 主流程 ==========
left_f32 = preprocess(in_left)
right_f32 = preprocess(in_right)

left_u8 = quantize_to_uint8(left_f32, scale_left, zp_left, verbose=True)
right_u8 = quantize_to_uint8(right_f32, scale_right, zp_right, verbose=True)

# 保存二进制
left_u8.tofile(os.path.join(out_dir, "left_u8.bin"))
right_u8.tofile(os.path.join(out_dir, "right_u8.bin"))

# 保存调试 numpy 文件
np.save(os.path.join(out_dir, "left_u8.npy"), left_u8.astype(np.float32))
np.save(os.path.join(out_dir, "right_u8.npy"), right_u8.astype(np.float32))

print("✅ uint8 量化完成")
print(f"Left shape = {left_u8.shape}, Right shape = {right_u8.shape}")






