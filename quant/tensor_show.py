import numpy as np
from PIL import Image
import sys
import cv2

def txt_to_png(txt_file_path, output_png_path, width=512, height=256):
    # 读取文本文件
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    
    # 转换为浮点数数组
    disp_values = [float(line.strip()) for line in lines]
    
    # 检查数据量是否匹配
    expected_size = width * height
    if len(disp_values) != expected_size:
        print(f"警告：数据量({len(disp_values)})与图像尺寸({expected_size})不匹配")
        # 如果数据不足，用0填充；如果过多，截断
        if len(disp_values) < expected_size:
            disp_values.extend([0.0] * (expected_size - len(disp_values)))
        else:
            disp_values = disp_values[:expected_size]
    
    # 转换为numpy数组并重塑为图像尺寸
    disp_array = np.array(disp_values).reshape(height, width).astype(np.float32)
    # disp_array.tofile("board.bin")
    np.save('./output/quant/disparity.npy', disp_array)
    
    
    
    print(disp_array)   
    print(disp_array.dtype)   
    print(disp_array.shape)   
    
    
    cv2.imwrite('./output/quant/disparity.png', disp_array)
    
    # 归一化到0-255范围（灰度图像）
    disp_min = np.min(disp_array)
    disp_max = np.max(disp_array)
    
   
    print(f"max is {disp_max} ; min is {disp_min}")
    
    if disp_max > disp_min:
        # 归一化并缩放到0-255
        disp_normalized = (disp_array - disp_min) / (disp_max - disp_min) * 255
    else:
        # 如果所有值相同，设为中灰色
        disp_normalized = np.full_like(disp_array, 128)
         
    
    # 转换为8位无符号整数
    disp_uint8 = disp_normalized.astype(np.uint8)
    
    # 创建PIL图像并保存
    img = Image.fromarray(disp_uint8, mode='L')
    img.save(output_png_path)
    print(f"图像已保存到：{output_png_path}")
    
    
    disp_vis = cv2.normalize(disp_array, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)
    cv2.imshow("Disparity Map", disp_vis)
    
    while True:
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python script.py <输入txt文件> <输出png文件>")
        sys.exit(1)
    
    input_txt = sys.argv[1]
    output_png = sys.argv[2]
    
    txt_to_png(input_txt, output_png)
