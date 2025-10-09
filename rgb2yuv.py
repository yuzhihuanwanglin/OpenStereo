#!/usr/bin/env python3
"""
PNG/JPG -> NV12 / NV21 / YUV420 工具
用法:
    python img2yuv.py input.png output.nv12 --format NV12 --standard BT709
参数:
    --format : NV12 | NV21 | I420
    --standard : BT709 | BT601
注意:
    输出文件为原始二进制 YUV，可直接用于 NPU 输入
"""

import cv2, numpy as np, argparse, pathlib

def rgb_to_yuv_planes(rgb, standard="BT709", full_range=True):
    """
    RGB (H,W,3) uint8 -> Y, U, V planes
    """
    rgb = rgb.astype(np.float32)
    if standard.upper() == "BT709":
        coeffs = {
            "Kr": 0.2126, "Kg": 0.7152, "Kb": 0.0722
        }
    else:  # BT601
        coeffs = {
            "Kr": 0.299, "Kg": 0.587, "Kb": 0.114
        }
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]

    # Full / Limited Range
    if full_range:
        Y = coeffs["Kr"]*R + coeffs["Kg"]*G + coeffs["Kb"]*B
        U = 0.5*(B - Y) + 128
        V = 0.5*(R - Y) + 128
    else:
        # Limited 16~235 / 16~240
        Y = 16 + (219)*(coeffs["Kr"]*R + coeffs["Kg"]*G + coeffs["Kb"]*B)/255
        U = 128 + 112*(0.5*(B - Y))/255
        V = 128 + 112*(0.5*(R - Y))/255

    Y = np.clip(Y,0,255).astype(np.uint8)
    U = np.clip(U,0,255).astype(np.uint8)
    V = np.clip(V,0,255).astype(np.uint8)
    return Y, U, V

def resize_even(img, w, h):
    # NV12/NV21/I420 要求偶数
    return cv2.resize(img, (w//2*2, h//2*2), interpolation=cv2.INTER_AREA)

def rgb_to_yuv420(rgb, format_type="I420", standard="BT709", full_range=True):
    """
    RGB -> NV12 / NV21 / I420
    """
    h, w = rgb.shape[:2]
    Y, U, V = rgb_to_yuv_planes(rgb, standard, full_range)

    # 下采样 U/V 2x2 -> H/2 x W/2
    U_sub = cv2.resize(U, (w//2,h//2), interpolation=cv2.INTER_AREA)
    V_sub = cv2.resize(V, (w//2,h//2), interpolation=cv2.INTER_AREA)

    if format_type.upper() == "I420":
        # Planar: YYYYYY UUUU VVVV
        return np.vstack([Y, U_sub, V_sub])
    elif format_type.upper() == "NV12":
        # Semi-planar: Y + interleaved UV
        UV = np.zeros((h//2,w), dtype=np.uint8)
        UV[:,0::2] = U_sub
        UV[:,1::2] = V_sub
        return np.vstack([Y, UV])
    elif format_type.upper() == "NV21":
        # Semi-planar: Y + interleaved VU
        VU = np.zeros((h//2,w), dtype=np.uint8)
        VU[:,0::2] = V_sub
        VU[:,1::2] = U_sub
        return np.vstack([Y, VU])
    else:
        raise ValueError("Unsupported format_type")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input PNG/JPG")
    parser.add_argument("output", help="Output YUV binary file")
    parser.add_argument("--format", default="NV12", choices=["NV12","NV21","I420"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--standard", default="BT709", choices=["BT709","BT601"])
    parser.add_argument("--full_range", action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read {args.input}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize
    if args.width and args.height:
        img = resize_even(img, args.width, args.height)
    else:
        img = resize_even(img, img.shape[1], img.shape[0])

    yuv = rgb_to_yuv420(img, args.format, args.standard, args.full_range)
    pathlib.Path(args.output).write_bytes(yuv.tobytes())
    print(f"{args.format} {args.standard}_{'FULL' if args.full_range else 'LIMIT'} saved: {args.output}, size={yuv.nbytes} bytes")

if __name__ == "__main__":
    main()
