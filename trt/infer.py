import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import json
import time
import os

# ------------------------
# 1. 初始化 TensorRT Runtime
# ------------------------
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
engine_path = "/home/wanglin/workspace/logicAI/onnx/light_fp32.trt"

assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
print(f"[INFO] TensorRT version: {trt.__version__}")
print(f"[INFO] Loading TensorRT engine: {engine_path}")

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
stream = cuda.Stream()

# ------------------------
# 2. 设置输入形状（等价于 trtexec --shapes）
# ------------------------
# TensorRT 10.x: 使用 set_input_shape()
context.set_input_shape("left_img", (1, 3, 256, 512))
context.set_input_shape("right_img", (1, 3, 256, 512))

# 确认 shape 已生效
print("\n[INFO] Input/Output bindings:")
for name in engine:
    io_mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = context.get_tensor_shape(name)
    print(f"  {name:<20} | {'Input' if io_mode == trt.TensorIOMode.INPUT else 'Output':<6} | {dtype} | {shape}")

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
# 4. 准备输入数据
# ------------------------
for name, buf in bindings.items():
    if buf["is_input"]:
        img = np.random.randn(*buf["shape"]).astype(buf["dtype"])
        np.copyto(buf["host"], img.ravel())

# ------------------------
# 5. 设置 Tensor 地址绑定
# ------------------------
for name, buf in bindings.items():
    context.set_tensor_address(name, int(buf["device"]))

# ------------------------
# 6. 执行推理（等价于 trtexec --dumpProfile）
# ------------------------
n_iterations = 100
latencies = []
print("\n[INFO] Running inference ...")

for i in range(n_iterations):
    start = time.time()

    # 拷贝输入到 GPU
    for name, buf in bindings.items():
        if buf["is_input"]:
            cuda.memcpy_htod_async(buf["device"], buf["host"], stream)

    # 执行推理 (TensorRT 10 使用 execute_async_v3)
    context.execute_async_v3(stream_handle=stream.handle)

    # 拷贝输出回 CPU
    for name, buf in bindings.items():
        if not buf["is_input"]:
            cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)

    stream.synchronize()
    end = time.time()
    latencies.append((end - start) * 1000)

avg_latency = np.mean(latencies)
print(f"[INFO] Average latency over {n_iterations} runs: {avg_latency:.3f} ms")

# ------------------------
# 7. 导出输出结果（二进制）
# ------------------------
for name, buf in bindings.items():
    if not buf["is_input"]:
        out_file = f"{name}_output.bin"
        buf["host"].tofile(out_file)
        print(f"[INFO] Output saved to {out_file}")

# ------------------------
# 8. 导出 Profile 数据（JSON）
# ------------------------
profile_data = {
    "engine": engine_path,
    "avg_latency_ms": float(avg_latency),
    "bindings": {
        name: {
            "shape": tuple(map(int, buf["shape"])),
            "dtype": str(buf["dtype"]),
            "is_input": bool(buf["is_input"]),
        }
        for name, buf in bindings.items()
    },
}

with open("trt_profile.json", "w") as f:
    json.dump(profile_data, f, indent=2)

print("[INFO] Profiling data saved to trt_profile.json")
