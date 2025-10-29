import numpy as np
import argparse
import os

def compare_layers(board_file='./output/quant/disparity.npy',
                   onnx_file='./output/export_disparity.npy',
                   threshold=1e-3):
    """
    对比两个层输出的相似度（余弦相似度 + 欧式距离指标）

    参数:
        board_file: 板端结果 .npy 文件路径
        onnx_file:  ONNX 模型结果 .npy 文件路径
        threshold:  差异阈值 (默认 1e-3)
    """
    if not os.path.exists(board_file):
        raise FileNotFoundError(f"找不到文件: {board_file}")
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"找不到文件: {onnx_file}")

    # 加载数据
    board_npy = np.load(board_file)
    onnx_npy = np.load(onnx_file)

    print(f"\n✅ 已加载文件:")
    print(f"  Board: {board_file}  shape={board_npy.shape}")
    print(f"  ONNX : {onnx_file}  shape={onnx_npy.shape}")

    # 展平
    board_flat = board_npy.flatten().astype(np.float32)
    onnx_flat = onnx_npy.flatten().astype(np.float32)

    # -------- 计算余弦相似度 --------
    num = np.dot(board_flat, onnx_flat)
    demon = np.linalg.norm(board_flat) * np.linalg.norm(onnx_flat)
    cosine_similarity = num / demon if demon != 0 else 0.0

    # -------- 计算欧式距离和误差指标 --------
    diff = np.abs(board_flat - onnx_flat)
    euclidean_distance = np.linalg.norm(diff)
    max_error = diff.max()
    mean_error = diff.mean()
    ratio_exceed = np.mean(diff > threshold)
    top5 = np.sort(diff)[-10:][::-1]   # 从大到小排序取前5个

    # -------- 输出结果 --------
    print(f"\n📊 对比结果：")
    print(f"余弦相似度: {cosine_similarity:.6f}")
    print(f"欧式距离: {euclidean_distance:.6f}")
    
    print(f"平均绝对误差: {mean_error:.6e}")
    print(f"最大绝对误差top5: {top5}")
    
    print(f"差异超过阈值({threshold:.0e})的比例: {ratio_exceed:.2%}\n")
    
    
    epsilon = 1e-8
    percent_diff = diff / (np.abs(onnx_flat) + epsilon) * 100

    mean_percent_loss = np.mean(percent_diff)
    max_percent_loss = np.max(percent_diff)
    
    top5 = np.sort(percent_diff)[-10:][::-1]   # 从大到小排序取前5个

    print(f"平均百分比损失: {mean_percent_loss:.4f}%")
    print(f"最大百分比损失top5: {top5}%")




'''
python quant/disparity_diff.py --board ./output/quant/disparity.npy --onnx ./output/export_disparity.npy

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较两个层输出文件 (.npy)")
    parser.add_argument("--board", default="./output/quant/disparity.npy",
                        help="板端输出文件路径 (.npy)")
    parser.add_argument("--onnx", default="./output/export_disparity.npy",
                        help="ONNX 模型输出文件路径 (.npy)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="误差阈值 (默认 1e-3)")
    args = parser.parse_args()

    compare_layers(args.board, args.onnx, args.threshold)
