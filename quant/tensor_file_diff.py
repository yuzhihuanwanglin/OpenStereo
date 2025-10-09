import pandas as pd
import numpy as np
import os

# 输入 CSV 文件路径
input_file = "/home/lc/share/gaoshantest/compare_light_out/compare_result.csv"



# 读取 CSV
df = pd.read_csv(input_file)

# 提取第二列和第三列
pairs = list(zip(df.iloc[:, 1], df.iloc[:, 2]))

for layer, node in pairs:
    # print(layer, " -> ", node)
    
    npy_file = '/home/lc/share/gaoshantest/compare_light_out/layerdump/' + layer + '_quanted.npy'
    bin_file = '/home/lc/share/gaoshantest/model_exec_out/TENSOR_01/tensor/' + node


    if os.path.exists(npy_file) == False:
        print(f"{npy_file} 文件不存在")
        continue
        
    if os.path.exists(bin_file) == False:
        print(f"{bin_file} 文件不存在")
        continue
        
    arr_bin = np.fromfile(bin_file, dtype=np.int8).astype(np.float32)
    # print(arr_bin)


    arr_npy = np.load(npy_file)
    # print(arr_npy)

    arr_bin_flat = arr_bin.flatten()
    arr_npy_flat = arr_npy.flatten()

    num = float(np.dot(arr_bin_flat,arr_npy_flat))
    demon = np.linalg.norm(arr_bin_flat)*np.linalg.norm(arr_npy_flat)

    print(f'{npy_file} == {bin_file}: {num/demon}')
