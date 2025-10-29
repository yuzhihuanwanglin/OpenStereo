#!/usr/bin/env python3
import os
import glob
import time
import json
import math
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ------------------------
# 配置
# ------------------------
engine_paths = {
    "FP32": "/home/wanglin/workspace/logicAI/onnx/light_fp32.trt",
    "FP16": "/home/wanglin/workspace/logicAI/onnx/light_fp16.trt",
    "INT8": "/home/wanglin/workspace/logicAI/onnx/light_int8.trt"
}

# left_dir = "/media/wanglin/Elements/datasets/SceneFlow/Driving/frames_cleanpass/15mm_focallength/scene_backwards/fast/left"
# right_dir = "/media/wanglin/Elements/datasets/SceneFlow/Driving/frames_cleanpass/15mm_focallength/scene_backwards/fast/right"

# left_dir = "/media/wanglin/Elements/datasets/KITTI2015/training/image_2"
# right_dir = "/media/wanglin/Elements/datasets/KITTI2015/training/image_3"


left_dir = "/media/wanglin/Elements/datasets/DrivingStereo/cloudy/left-image-full-size/cloudy/left-image-full-size"
right_dir = "/media/wanglin/Elements/datasets/DrivingStereo/cloudy/right-image-full-size/cloudy/right-image-full-size"


disp_out_dir = "./trt/npy"
N = 10  # 只取前 N 张

input_shapes = {
    "left_img": (1, 3, 256, 512),
    "right_img": (1, 3, 256, 512)
}

N_WARMUP = 5
N_RUNS = 50

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ------------------------
# 工具函数
# ------------------------
def mb(x): return x / 1024.0 / 1024.0

def safe_load_engine(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, "Engine file missing or empty", None
    free_before, _ = cuda.mem_get_info()
    try:
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None, "Deserialize returned None", None
    except Exception as e:
        return None, f"Exception: {repr(e)}", None
    free_after, _ = cuda.mem_get_info()
    return engine, None, mb(free_before - free_after)

def preprocess_image(img_path, shape):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((shape[3], shape[2]))  # W,H
    img = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)  # 1xCxHxW
    return img

def prepare_buffers_and_bind(context, engine):
    buffers = {}
    for name in engine:
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        shape = context.get_tensor_shape(name)
        size = int(trt.volume(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        buffers[name] = {
            "host": host_mem,
            "device": device_mem,
            "shape": shape,
            "dtype": dtype,
            "is_input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        }
        context.set_tensor_address(name, int(device_mem))
    return buffers

def run_inference(context, buffers, stream, n_runs=N_RUNS):
    latencies = []
    for _ in range(N_WARMUP):
        for name, buf in buffers.items():
            if buf["is_input"]:
                cuda.memcpy_htod_async(buf["device"], buf["host"], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for name, buf in buffers.items():
            if not buf["is_input"]:
                cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)
        stream.synchronize()
    # 正式测
    for _ in range(n_runs):
        start = time.time()
        for name, buf in buffers.items():
            if buf["is_input"]:
                cuda.memcpy_htod_async(buf["device"], buf["host"], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for name, buf in buffers.items():
            if not buf["is_input"]:
                cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)
        stream.synchronize()
        end = time.time()
        latencies.append((end - start) * 1000.0)
    avg_latency = float(np.mean(latencies))
    outputs = {}
    for name, buf in buffers.items():
        if not buf["is_input"]:
            arr = np.frombuffer(buf["host"], dtype=buf["dtype"]).reshape(buf["shape"]).astype(np.float32)
            outputs[name] = arr
    return avg_latency, outputs

def compute_metrics(ref, other, top_k=10):
    ref_flat = ref.flatten()
    other_flat = other.flatten()

    # 余弦相似度
    cos_sim = float(np.dot(ref_flat, other_flat) / 
                    (np.linalg.norm(ref_flat) * np.linalg.norm(other_flat) + 1e-12))

    # RMSE
    diff = ref_flat - other_flat
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # 误差最大的前 top_k 个像素索引和值
    abs_diff = np.abs(diff)
    top_indices = np.argsort(-abs_diff)[:top_k]  # 误差从大到小排序
    top_errors = abs_diff[top_indices]

    # 转换成二维索引，如果 ref 是二维图像
    if ref.ndim == 2:
        h, w = ref.shape
        top_coords = [(int(idx // w), int(idx % w)) for idx in top_indices]
    else:
        top_coords = [(int(idx)) for idx in top_indices]

    top10 = [{"coord": coord, "error": float(err)} for coord, err in zip(top_coords, top_errors)]

    return {"cos_sim": cos_sim, "RMSE": rmse, "top10_errors": top10}


# ------------------------
# 批量推理准备
# ------------------------
left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))[:N]
right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))[:N]
disp_outputs = [os.path.join(disp_out_dir, os.path.basename(f).replace(".png", "_disp.npy"))
                for f in left_images]

assert len(left_images) == len(right_images) == len(disp_outputs), "左右图数量不一致"

# ------------------------
# 主循环：FP32/FP16/INT8
# ------------------------
results = {}
reference_outputs = None
reference_name = None

for mode, engine_path in engine_paths.items():
    print(f"\n[INFO] Processing {mode} engine: {engine_path}")
    engine, err, mem_used_mib = safe_load_engine(engine_path)
    if engine is None:
        print(f"  [ERROR] Load failed: {err}")
        results[mode] = {"path": engine_path, "error": err}
        continue

    context = engine.create_execution_context()
    stream = cuda.Stream()
    buffers = prepare_buffers_and_bind(context, engine)

    total_latency = 0.0
    for idx, (l_img, r_img, out_path) in enumerate(zip(left_images, right_images, disp_outputs)):
        # 预处理
        buffers["left_img"]["host"][:] = preprocess_image(l_img, input_shapes["left_img"]).ravel()
        buffers["right_img"]["host"][:] = preprocess_image(r_img, input_shapes["right_img"]).ravel()

        # 推理
        avg_ms, outputs = run_inference(context, buffers, stream, n_runs=1)
        total_latency += avg_ms

        # 保存输出
        np.save(out_path.replace(".npy", f"_{mode}.npy"), outputs["disp_pred"])

        # 作为 FP32 参考输出
       # FP32 批量保存参考输出
        if mode == "FP32":
            if reference_outputs is None:
                reference_outputs = []
            reference_outputs.append(outputs["disp_pred"])
        else:
            # 精度对比，按图片索引对应
            metrics = compute_metrics(reference_outputs[idx], outputs["disp_pred"])
            top10_str = " ".join([f"{e['coord']}:{e['error']:.4f}" for e in metrics['top10_errors']])
            print(f"  [{idx+1}/{N}] {mode} metrics vs FP32: cos={metrics['cos_sim']:.6f}, RMSE={metrics['RMSE']:.6f}, top10_errors=[{top10_str}]")

    avg_latency_overall = total_latency / N
    fps = 1000.0 / avg_latency_overall if avg_latency_overall > 0 else float("inf")
    print(f"[INFO] {mode} engine avg latency per image: {avg_latency_overall:.3f} ms, FPS: {fps:.2f}, GPU mem used: {mem_used_mib:.2f} MiB")

    results[mode] = {
        "path": engine_path,
        "avg_latency_ms": avg_latency_overall,
        "throughput_fps": fps,
        "mem_used_mib": mem_used_mib
    }

# ------------------------
# 保存 JSON 报告
# ------------------------
with open("trt_batch_compare_report.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Batch inference report saved to trt_batch_compare_report.json")
