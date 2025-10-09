import numpy as np
import cv2
import os
import argparse

def check_nv12_file(nv12_file, width, height):
    """
    检查 NV12 文件格式与 Full/Limited Range，并经验判断 BT601/BT709
    """
    if not os.path.exists(nv12_file):
        raise FileNotFoundError(f"文件不存在: {nv12_file}")
    
    # 读取文件
    with open(nv12_file,'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    expected_size = width * height * 3 // 2
    if len(data) != expected_size:
        raise ValueError(f"❌ 文件大小 {len(data)} 不匹配 NV12 期望 {expected_size}")
    
    # 分离 Y 和 UV
    y = data[:width*height]
    uv = data[width*height:]
    
    y_min, y_max = y.min(), y.max()
    uv_min, uv_max = uv.min(), uv.max()
    
    # Full / Limited Range 判断
    if (y_min >= 16 and y_max <= 235) and (uv_min >= 16 and uv_max <= 240):
        full_range = False
    else:
        full_range = True
    
    # 经验判断 BT601/BT709
    y_mean = y.mean()
    bt = 'BT709'#'BT601' if y_mean < 128 else 'BT709'
    
    print("✅ NV12 文件检查通过")
    print(f"Y 通道范围: {y_min}~{y_max}, UV 范围: {uv_min}~{uv_max}")
    print("Full Range" if full_range else "Limited Range")
    print(f"BT 色彩空间（经验判断）: {bt} (非完全精确)")
    
    return data, full_range, bt

def nv12_to_png(nv12_file, png_file, width, height, full_range=True, bt='BT601'):
    data, _, _ = check_nv12_file(nv12_file, width, height)
    
    # 分离 Y 和 UV
    y = data[:width*height].reshape((height, width))
    uv = data[width*height:].reshape((height//2, width))  # 修正这里
    
    u = uv[:, 0::2]
    v = uv[:, 1::2]
    
    # 上采样
    u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Full / Limited Range
    if not full_range:
        y = (y.astype(np.float32) - 16) * (255/219)
        u = (u.astype(np.float32) - 128) * (255/224)
        v = (v.astype(np.float32) - 128) * (255/224)
    else:
        y = y.astype(np.float32)
        u = u.astype(np.float32) - 128
        v = v.astype(np.float32) - 128
    
    # BT601 / BT709 转 RGB
    if bt.upper() == 'BT601':
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
    else:
        r = y + 1.5748 * v
        g = y - 0.187324 * u - 0.468124 * v
        b = y + 1.8556 * u
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
    cv2.imwrite(png_file, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"✅ NV12 转 PNG 保存成功: {png_file}")


# -----------------------------
# 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="NV12 文件检查并转换为 PNG")
    parser.add_argument("--nv12_file", type=str, required=True, help="输入 NV12 文件路径")
    parser.add_argument("--png_file", type=str, required=True, help="输出 PNG 文件路径")
    parser.add_argument("--width", type=int, required=True, help="图像宽度")
    parser.add_argument("--height", type=int, required=True, help="图像高度")
    
    args = parser.parse_args()
    
    data, full_range, bt = check_nv12_file(args.nv12_file, args.width, args.height)
    nv12_to_png(args.nv12_file, args.png_file, args.width, args.height, full_range, bt)

if __name__ == "__main__":
    main()
