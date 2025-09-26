
# 对比compare生成的layer和  vs_model_exec -y生成的layer输出的余弦值 

import numpy as np
# npy_file = '/home/wanglin/tensors/npy/input_quanted.npy'
# bin_file = '/home/wanglin/tensors/tensor/Conv_0'
#0.9997243161988774

# npy_file = '/home/wanglin/tensors/npy/input.20_quanted.npy'
# bin_file = '/home/wanglin/tensors/tensor/LeakyRelu_9'
#0.9999412813751851


# npy_file = '/home/wanglin/tensors/npy/input.24_quanted.npy'
# bin_file = '/home/wanglin/tensors/tensor/Conv_10'


npy_file = '/home/wanglin/tensors/npy/disp_pred_quanted.npy'
bin_file = '/home/wanglin/tensors/tensor/ReduceSum_1138'

arr_bin = np.fromfile(bin_file, dtype=np.int8).astype(np.float32)
print(arr_bin)


arr_npy = np.load(npy_file)
print(arr_npy)

arr_bin_flat = arr_bin.flatten()
arr_npy_flat = arr_npy.flatten()

num = float(np.dot(arr_bin_flat,arr_npy_flat))
demon = np.linalg.norm(arr_bin_flat)*np.linalg.norm(arr_npy_flat)

print(num/demon)