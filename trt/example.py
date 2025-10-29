# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import json
# import time
# import os
# from PIL import Image

# # ------------------------
# # 配置
# # ------------------------
# engine_path = "/home/wanglin/workspace/logicAI/onnx/light_fp32.trt"
# left_img_path = "/home/wanglin/workspace/backup/OpenStereo/mgz/datas/left.png"
# right_img_path = "/home/wanglin/workspace/backup/OpenStereo/mgz/datas/right.png"
# output_disp_path = "trt/disp_pred.npy"

# # ------------------------
# # 1. 初始化 TensorRT Runtime
# # ------------------------
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
# print(f"[INFO] TensorRT version: {trt.__version__}")
# print(f"[INFO] Loading TensorRT engine: {engine_path}")

# with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#     engine = runtime.deserialize_cuda_engine(f.read())

# context = engine.create_execution_context()
# stream = cuda.Stream()

# # ------------------------
# # 2. 设置输入形状
# # ------------------------
# input_shapes = {
#     "left_img": (1, 3, 256, 512),
#     "right_img": (1, 3, 256, 512),
# }

# for name, shape in input_shapes.items():
#     context.set_input_shape(name, shape)

# # ------------------------
# # 3. 分配内存缓冲区
# # ------------------------
# bindings = {}
# for name in engine:
#     dtype = trt.nptype(engine.get_tensor_dtype(name))
#     shape = context.get_tensor_shape(name)
#     size = int(trt.volume(shape))
#     host_mem = cuda.pagelocked_empty(size, dtype)
#     device_mem = cuda.mem_alloc(host_mem.nbytes)

#     bindings[name] = {
#         "host": host_mem,
#         "device": device_mem,
#         "shape": shape,
#         "dtype": dtype,
#         "is_input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
#     }

# # ------------------------
# # 4. 从文件读取左右图片
# # ------------------------
# def load_and_preprocess(path, target_shape=(256, 512)):
#     img = Image.open(path).convert("RGB")
#     img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
#     arr = np.array(img).astype(np.float32)
#     # HWC -> CHW
#     arr = arr.transpose(2, 0, 1)
#     # 添加 batch 维度
#     arr = np.expand_dims(arr, axis=0)
#     # 归一化到 [0,1] 或按模型要求
#     arr /= 255.0
#     return arr

# left_img = load_and_preprocess(left_img_path, input_shapes["left_img"][2:])
# right_img = load_and_preprocess(right_img_path, input_shapes["right_img"][2:])

# np.copyto(bindings["left_img"]["host"], left_img.ravel())
# np.copyto(bindings["right_img"]["host"], right_img.ravel())

# # ------------------------
# # 5. 设置 Tensor 地址绑定
# # ------------------------
# for name, buf in bindings.items():
#     context.set_tensor_address(name, int(buf["device"]))

# # ------------------------
# # 6. 执行推理
# # ------------------------
# start = time.time()

# for name, buf in bindings.items():
#     if buf["is_input"]:
#         cuda.memcpy_htod_async(buf["device"], buf["host"], stream)

# context.execute_async_v3(stream_handle=stream.handle)

# for name, buf in bindings.items():
#     if not buf["is_input"]:
#         cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)

# stream.synchronize()
# end = time.time()
# latency_ms = (end - start) * 1000
# print(f"[INFO] Inference latency: {latency_ms:.3f} ms")

# # ------------------------
# # 7. 保存输出视差图
# # ------------------------
# for name, buf in bindings.items():
#     if not buf["is_input"]:
#         out_arr = np.frombuffer(buf["host"], dtype=np.dtype(buf["dtype"]))
#         out_arr = out_arr.reshape(buf["shape"]).astype(np.float32)
#         np.save(output_disp_path, out_arr)
#         print(f"[INFO] Disparity map saved to {output_disp_path}")




#!/usr/bin/env python3
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

# ------------------------
# 配置
# ------------------------
engine_path = "/home/wanglin/workspace/logicAI/onnx/light_int8.trt"
left_img_path = "/home/wanglin/workspace/backup/OpenStereo/mgz/datas/left.png"
right_img_path = "/home/wanglin/workspace/backup/OpenStereo/mgz/datas/right.png"
output_disp_path = "trt/disp_pred.npy"
input_shapes = {"left_img": (1, 3, 256, 512),
                "right_img": (1, 3, 256, 512)}
N_WARMUP = 5
N_RUNS = 100

# ------------------------
# 1. 初始化 TensorRT Runtime
# ------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
print(f"[INFO] TensorRT version: {trt.__version__}")
print(f"[INFO] Loading TensorRT engine: {engine_path}")

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
stream = cuda.Stream()

# ------------------------
# 2. 设置输入形状
# ------------------------
for name, shape in input_shapes.items():
    context.set_input_shape(name, shape)

# ------------------------
# 3. 分配内存缓冲区
# ------------------------
bindings = {}
for name in engine:
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = context.get_tensor_shape(name)
    size = int(trt.volume(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings[name] = {
        "host": host_mem,
        "device": device_mem,
        "shape": shape,
        "dtype": dtype,
        "is_input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
    }

# ------------------------
# 4. 读取左右图像
# ------------------------
def preprocess_image(img_path, shape):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # HWC, BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[3], shape[2]))  # WH -> shape (N,C,H,W)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    img = np.expand_dims(img, axis=0)  # N,C,H,W
    return img

bindings["left_img"]["host"][:] = preprocess_image(left_img_path, input_shapes["left_img"]).ravel()
bindings["right_img"]["host"][:] = preprocess_image(right_img_path, input_shapes["right_img"]).ravel()

# ------------------------
# 5. 设置 tensor 地址绑定
# ------------------------
for name, buf in bindings.items():
    context.set_tensor_address(name, int(buf["device"]))

# ------------------------
# 6. 推理（使用 CUDA Event 精确测量）
# ------------------------
# 拷贝输入到 GPU
for name, buf in bindings.items():
    if buf["is_input"]:
        cuda.memcpy_htod_async(buf["device"], buf["host"], stream)

# warmup
for _ in range(N_WARMUP):
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

# 测量
latencies = []
start_event = cuda.Event()
end_event = cuda.Event()

for _ in range(N_RUNS):
    start_event.record(stream)
    context.execute_async_v3(stream_handle=stream.handle)
    end_event.record(stream)
    end_event.synchronize()
    latencies.append(start_event.time_till(end_event))  # ms

avg_latency = np.mean(latencies)
fps = 1000.0 / avg_latency if avg_latency > 0 else float("inf")
print(f"[INFO] Inference latency: {avg_latency:.3f} ms | FPS: {fps:.2f}")

# ------------------------
# 7. 拷回输出并保存为 npy
# ------------------------
for name, buf in bindings.items():
    if not buf["is_input"]:
        cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)
        stream.synchronize()
        out_arr = np.frombuffer(buf["host"], dtype=buf["dtype"]).reshape(buf["shape"]).astype(np.float32)
        np.save(output_disp_path, out_arr)
        print(f"[INFO] Disparity map saved to {output_disp_path}")
