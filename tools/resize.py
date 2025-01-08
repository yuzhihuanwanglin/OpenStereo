import cv2
import glob
import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import time  # 添加以模拟工作量并测试剩余时间显示
import argparse

def resize_image(input_path, input_path_replace,output_path_replace,target_size=(768, 384)):
    try:
        img = cv2.imread(input_path)
        if img is None:
            print("file error:", input_path)
            return
    except:
        print("no file:", input_path)
        return

    output_path = input_path.replace(input_path_replace, output_path_replace)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_height, original_width = img.shape[:2]

    # 计算缩放比例
    scale = max(target_size[1] / original_height, target_size[0] / original_width)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 确保 new_width 或 new_height 等于目标尺寸
    if new_width < target_size[0]:
        new_width = target_size[0]
        new_height = int(original_height * (new_width / original_width))  # 重新计算高度
    elif new_height < target_size[1]:
        new_height = target_size[1]
        new_width = int(original_width * (new_height / original_height))  # 重新计算宽度

    resized_img = cv2.resize(img, (new_width, new_height))

    # 保存缩放后的图像
    cv2.imwrite(output_path, resized_img)


def resize_all_image_paths(all_img_txt,input_path_replace,output_path_replace):
    # 使用 ThreadPoolExecutor 并行处理文件，并显示剩余时间
    with open(all_img_txt, 'r') as f:
        input_files = sorted([file.strip() for file in f.readlines()])

    with ThreadPoolExecutor() as executor:
        with tqdm.tqdm(total=len(input_files), desc="Resizing images", unit="img") as pbar:
            for _ in executor.map(lambda img_path: resize_image(img_path, input_path_replace,output_path_replace),input_files):
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all image paths from a text file.")
    parser.add_argument('input_path_file', type=str, help='Path to the text file containing image paths')
    parser.add_argument('input_path_replace', type=str, help='input image root path')
    parser.add_argument('output_path_replace', type=str, help='output image root path')
    
    args = parser.parse_args()

    all_img_txt = args.input_path
    input_path_replace = args.input_path_replace
    output_path_replace = args.output_path_replace

    resize_all_image_paths(all_img_txt,input_path_replace,output_path_replace)

# python resize_image.py --input_path_file ./xxx.txt --input_path_replace /mnt/public_data/imagenet21k --output_path_replace /mnt/public_data/imagenet21k_resize


