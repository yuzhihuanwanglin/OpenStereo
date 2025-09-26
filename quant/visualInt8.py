# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# @file: int8_input_debug.py
# @brief: 从 NPU int8 bin 文件读取输入，反量化为 float32，并可视化。可选与 FP32 输入对比。
# """

# import numpy as np
# import matplotlib.pyplot as plt

# # ================================
# # 配置参数
# # ================================
# INPUT_SHAPE = (1, 3, 256, 512)  # (N,C,H,W)
# NORMALIZE_MEAN = [0.485, 0.456, 0.406]
# NORMALIZE_STD  = [0.229, 0.224, 0.225]

# # NPU int8 量化参数
# SCALE_LEFT  = 0.017425
# SCALE_RIGHT = 0.017288
# ZERO_POINT  = 0

# # 输入 bin 文件路径
# # LEFT_BIN_PATH  = "./left_img_input.bin"
# # RIGHT_BIN_PATH = "./right_img_input.bin"
# LEFT_BIN_PATH  = "./left.bin"
# RIGHT_BIN_PATH = "./right.bin"

# # ================================
# # 函数定义
# # ================================
# def load_int8_bin(file_path: str, shape=INPUT_SHAPE) -> np.ndarray:
#     """
#     从 bin 文件读取 int8 数据，并 reshape 成指定形状
#     """
#     data = np.fromfile(file_path, dtype=np.int8)
#     if data.size != np.prod(shape):
#         raise ValueError(f"Bin 文件大小 {data.size} 与目标 shape {shape} 不匹配")
#     return data.reshape(shape)

# def dequantize_int8(int8_tensor: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
#     """
#     将 int8 张量反量化为 float32
#     """
#     return (int8_tensor.astype(np.float32) - zero_point) * scale

# def visualize_image(float_tensor: np.ndarray, title: str = "Image"):
#     """
#     可视化 float32 图像，支持输入 (1,3,H,W) 或 (3,H,W)
#     已经经过 Normalize，显示前会反标准化到 [0,1]
#     """
#     if float_tensor.ndim == 4:
#         float_tensor = float_tensor[0]  # 去 batch 维度

#     mean = np.array(NORMALIZE_MEAN).reshape(3,1,1)
#     std  = np.array(NORMALIZE_STD).reshape(3,1,1)

#     img = float_tensor * std + mean
#     img = np.clip(img, 0, 1)
#     img = img.transpose(1,2,0)  # CHW -> HWC

#     print(img)

#     plt.figure(figsize=(6,4))
#     plt.imshow(img)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# def compare_with_fp32(int8_float: np.ndarray, fp32_tensor: np.ndarray, name: str = "Left"):
#     """
#     与 FP32 原始输入比较差异
#     """
#     if int8_float.shape != fp32_tensor.shape:
#         raise ValueError("Shape mismatch between int8 float and FP32 tensor")
#     diff = np.abs(int8_float - fp32_tensor)
#     print(f"[{name}] max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

# # ================================
# # 主流程
# # ================================
# def main():
#     # 1. 读取 int8 bin
#     left_int8  = load_int8_bin(LEFT_BIN_PATH)
#     right_int8 = load_int8_bin(RIGHT_BIN_PATH)
#     print("Loaded int8 tensors:")
#     print(f"  Left  : {left_int8.shape}, dtype={left_int8.dtype}")
#     print(f"  Right : {right_int8.shape}, dtype={right_int8.dtype}")

#     # 2. 反量化
#     left_float  = dequantize_int8(left_int8,  SCALE_LEFT,  ZERO_POINT)
#     right_float = dequantize_int8(right_int8, SCALE_RIGHT, ZERO_POINT)
#     print("Dequantized to float32:")
#     print(f"  Left  : {left_float.shape}, dtype={left_float.dtype}")
#     print(f"  Right : {right_float.shape}, dtype={right_float.dtype}")

#     # 3. 可视化
#     visualize_image(left_float,  "Left Image (Dequantized)")
#     visualize_image(right_float, "Right Image (Dequantized)")

#     # 4. 可选：与 FP32 原始输入比较
#     # 需要提前准备 fp32_left / fp32_right
#     # compare_with_fp32(left_float,  fp32_left,  "Left")
#     # compare_with_fp32(right_float, fp32_right, "Right")

# if __name__ == "__main__":
#     main()



















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: int8_bin_to_u8c3.py
@brief: 从 NPU int8 bin 文件读取输入，反量化为 float32，转换为 U8C3 RGB 图像并保存为 PNG
"""

import numpy as np
from PIL import Image
import os

# ================================
# 配置参数
# ================================
INPUT_SHAPE = (1, 3, 256, 512)  # (N,C,H,W)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# NPU int8 量化参数
SCALE_LEFT  = 0.017425
SCALE_RIGHT = 0.017288
ZERO_POINT  = 0

# 输入 bin 文件路径
LEFT_BIN_PATH  = "left_img_input.bin"
RIGHT_BIN_PATH = "right_img_input.bin"

# 输出 PNG 路径
LEFT_PNG_PATH  = "left_image_u8c3.png"
RIGHT_PNG_PATH = "right_image_u8c3.png"

# ================================
# 函数定义
# ================================
def load_int8_bin(file_path: str, shape=INPUT_SHAPE) -> np.ndarray:
    """
    从 bin 文件读取 int8 数据，并 reshape 成指定形状
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 不存在")
    data = np.fromfile(file_path, dtype=np.int8)
    if data.size != np.prod(shape):
        raise ValueError(f"Bin 文件大小 {data.size} 与目标 shape {shape} 不匹配")
    return data.reshape(shape)

def dequantize_int8(int8_tensor: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    """
    将 int8 张量反量化为 float32
    """
    return (int8_tensor.astype(np.float32) - zero_point) * scale

def int8_to_u8c3(int8_tensor: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    """
    将 int8 tensor 转换为 U8C3 RGB 图像
    int8_tensor: (1,3,H,W) 或 (3,H,W)
    返回: (H,W,3) uint8
    """
    # 反量化
    float_tensor = dequantize_int8(int8_tensor, scale, zero_point)

    # 去 batch 维度
    if float_tensor.ndim == 4:
        float_tensor = float_tensor[0]

    # 反标准化
    mean = np.array(NORMALIZE_MEAN).reshape(3,1,1)
    std  = np.array(NORMALIZE_STD).reshape(3,1,1)
    img = float_tensor * std + mean

    # clip 并映射到 0~255
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)

    # CHW -> HWC
    img = img.transpose(1,2,0)
    return img

def save_u8c3_image(img: np.ndarray, path: str):
    """
    保存 U8C3 RGB 图像为 PNG
    """
    Image.fromarray(img).save(path)
    print(f"Saved image to {path}, shape={img.shape}, dtype={img.dtype}")

# ================================
# 主流程
# ================================
def main():
    # 1. 读取 int8 bin
    left_int8  = load_int8_bin(LEFT_BIN_PATH)
    right_int8 = load_int8_bin(RIGHT_BIN_PATH)

    print("Loaded int8 tensors:")
    print(f"  Left  : {left_int8.shape}, dtype={left_int8.dtype}")
    print(f"  Right : {right_int8.shape}, dtype={right_int8.dtype}")

    # 2. 转换为 U8C3
    left_u8c3  = int8_to_u8c3(left_int8,  SCALE_LEFT,  ZERO_POINT)
    right_u8c3 = int8_to_u8c3(right_int8, SCALE_RIGHT, ZERO_POINT)

    # 3. 保存 PNG
    save_u8c3_image(left_u8c3, LEFT_PNG_PATH)
    save_u8c3_image(right_u8c3, RIGHT_PNG_PATH)

if __name__ == "__main__":
    main()

