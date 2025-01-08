import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import tqdm
import argparse

def read_one_img(img_path):
    try:
        a = cv2.imread(img_path)
        if a is None:
            print(img_path)
            return img_path
    except:
        print(img_path)
        return img_path

def read_all_images(input_txt,error_txt):
    with open(input_txt, 'r') as f:
        img_paths = [path.strip() for path in f.readlines()]
    print("start ",img_paths[0])
    with ThreadPoolExecutor() as executor:
        target_dir_list = tqdm.tqdm(executor.map(read_one_img, img_paths), total=len(img_paths))
        
    empty_paths = sorted([x for x in target_dir_list if x is not None])


    with open(error_txt, 'w') as f:
        for path in empty_paths:
            f.write(path + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check image read")
    parser.add_argument('input_path_file', type=str, help='Path to the text file containing image paths')
    parser.add_argument('error_path_dile', type=str, help='Path to the error files')
    
    args = parser.parse_args()

    input_txt = args.input_path_file
    error_txt = args.error_path_file

    read_all_images(input_txt,error_txt)

    

# python resize_image.py --input_path_file './google_landmarks.txt' --error_path_file './google_landmarks_error.txt'


