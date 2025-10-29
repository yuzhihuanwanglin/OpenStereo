import numpy as np
from PIL import Image
from torchvision import transforms
import os

# ========== 配置 ==========
in_left = "/home/fays007/lc/share/mgz/datas/left.png"
in_right = "/home/fays007/lc/share/mgz/datas/right.png"
out_dir = "output/quant/"
os.makedirs(out_dir, exist_ok=True)

input_size = (512,256)  # (W,H)
scale_left = 0.000073
scale_right = 0.000081
zp_left = 29172
zp_right = 2**15


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
    # 输出 (1, C, H, W)，且为 np.float32
    return np.expand_dims(t(img).numpy().astype(np.float32), axis=0)

# ========== 量化函数 (正确、稳健的 uint16 量化) ==========
def quantize_to_uint16(arr_float, scale, zero_point=None, verbose=True):
    """
    arr_float: numpy array, float32, 任意形状
    scale: 单个 float 标量（非零）
    zero_point: 如果 None，则默认使用 2**15 (32768)
    返回: uint16 numpy array（形状与输入相同）
    """
    if scale == 0 or zero_point is None:
        raise ValueError("scale must be non-zero")

    # 保证浮点类型
    arr = arr_float.astype(np.float64)  # 用高精度避免舍入误差累积
    print(arr.dtype, arr.min(), arr.max())

    

    # 量化步骤：先除以 scale，再四舍五入，再加 zero_point，再裁剪
    q = np.round(arr / float(scale)) + float(zero_point)

    # 裁剪到 uint16 范围
    q = np.clip(q, 0, 65535)

    q_u16 = q.astype(np.uint16)

    if verbose:
        # 打印一些验证信息
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        q_min, q_max = int(np.min(q_u16)), int(np.max(q_u16))
        z_map = int(np.round(0.0 / float(scale) + float(zero_point)))
        print(f"[quantize_to_uint16] float range: [{a_min:.6f}, {a_max:.6f}]")
        print(f"[quantize_to_uint16] uint16 range after quantize: [{q_min}, {q_max}]")
        print(f"[quantize_to_uint16] scale={scale}, zero_point={zero_point} -> 0.0 maps to {z_map}")

    return q_u16

# ========== 主流程 ==========
left_f32 = preprocess(in_left)   # shape (1,3,H,W)
right_f32 = preprocess(in_right)

left_u16 = quantize_to_uint16(left_f32, scale_left, zp_left, verbose=True)
right_u16 = quantize_to_uint16(right_f32, scale_right, zp_right, verbose=True)

# 保存原始 float32（numpy）
# np.save(os.path.join(out_dir, "left_32.npy"), left_f32)
# np.save(os.path.join(out_dir, "right_32.npy"), right_f32)

# 如果需要 little-endian 二进制（多数推理引擎期望小端），把 dtype 转为 '<u2'：
# left_u16_le = left_u16.astype('<u2')
# right_u16_le = right_u16.astype('<u2')

left_u16.tofile(os.path.join(out_dir, "left_u16.bin"))
right_u16.tofile(os.path.join(out_dir, "right_u16.bin"))

# 方便调试的 numpy 副本（将 uint16 以 float32 存储只是为了查看）
np.save(os.path.join(out_dir, "left_u16.npy"), left_u16.astype(np.float32))
np.save(os.path.join(out_dir, "right_u16.npy"), right_u16.astype(np.float32))

print("✅ uint16 量化完成")
print(f"Left shape = {left_u16.shape}, Right shape = {right_u16.shape}")
