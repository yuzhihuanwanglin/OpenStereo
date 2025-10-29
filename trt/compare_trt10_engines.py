#!/usr/bin/env python3
"""
TensorRT 10.x 多引擎性能与精度对比脚本
- 自动跳过无法反序列化的 engine 并记录错误
- 计算平均延迟 (ms) 与吞吐 (FPS)
- 记录加载时的显存占用 (MiB)
- 若第一个 engine 为基准 (FP32)，与后续 engine 比较 L1 / L2(RMSE) / PSNR
- 输出 JSON 报告 trt_compare_report.json
"""
import os
import time
import json
import math
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ---------- 配置 ----------
engine_paths = [
    "/home/wanglin/workspace/logicAI/onnx/light_fp32.trt",
    "/home/wanglin/workspace/logicAI/onnx/light_fp16.trt",
    "/home/wanglin/workspace/logicAI/onnx/light_int8.trt",
]
# 输入形状（与构建 engine 时相同）
input_shapes = {
    "left_img": (1, 3, 256, 512),
    "right_img": (1, 3, 256, 512)
}
N_RUNS = 100
# 若想用真实数据做精度评估，请在这里改写准备输入数据的逻辑
# ---------- end config ----------

def mb(x): return x / 1024.0 / 1024.0

def safe_load_engine(path):
    """加载 engine 并返回 engine/context, 同时测量加载前后显存差值 (MiB).
       若无法加载返回 (None, error_message, mem_before, mem_after)
    """
    if not os.path.exists(path):
        return None, f"Engine file not found: {path}", None, None
    size = os.path.getsize(path)
    if size == 0:
        return None, f"Engine file empty: {path}", None, None

    # 测显存（free, total）
    free_before, total = cuda.mem_get_info()
    try:
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            free_after, _ = cuda.mem_get_info()
            return None, "deserialize returned None (corrupt/unsupported engine)", mb(free_before - free_after), mb(total)
    except Exception as e:
        free_after, _ = cuda.mem_get_info()
        return None, f"Exception during deserialize: {repr(e)}", mb(free_before - free_after), mb(total)

    free_after, _ = cuda.mem_get_info()
    mem_used_mib = mb(free_before - free_after)
    return engine, None, mem_used_mib, mb(total)

def prepare_buffers_and_bind(context, engine, input_shapes):
    """为 engine/context 分配 host/device 缓冲，并绑定地址（TRT10 API）"""
    # 设置输入形状
    for name, shape in input_shapes.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)

    # 分配
    buffers = {}
    for name in engine:
        mode = engine.get_tensor_mode(name)
        dtype_trt = engine.get_tensor_dtype(name)
        dtype = np.dtype(trt.nptype(dtype_trt))   # ✅ 转换为 numpy.dtype
        shape = context.get_tensor_shape(name)
        size = int(trt.volume(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)  # ✅ 现
    # 绑定 device 地址
    for name, buf in buffers.items():
        context.set_tensor_address(name, int(buf["device"]))
    return buffers

def fill_inputs_random(buffers):
    for name, buf in buffers.items():
        if buf["is_input"]:
            arr = np.random.randn(*buf["shape"]).astype(np.float32)
            # 若 engine 输入 dtype 不是 float32，可 cast (we compare outputs in float32 later)
            arr = arr.astype(np.dtype(buf["dtype"]))
            np.copyto(buf["host"], arr.ravel())
    return

def run_engine(engine, input_shapes, n_runs=N_RUNS):
    """返回 avg_latency_ms, outputs_dict (name -> numpy array), mem_used_mib"""
    context = engine.create_execution_context()
    # prepare buffers
    buffers = prepare_buffers_and_bind(context, engine, input_shapes)
    # fill inputs
    fill_inputs_random(buffers)

    stream = cuda.Stream()
    latencies = []

    # warmup few runs (not counted)
    for _ in range(5):
        # copy inputs
        for name, buf in buffers.items():
            if buf["is_input"]:
                cuda.memcpy_htod_async(buf["device"], buf["host"], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for name, buf in buffers.items():
            if not buf["is_input"]:
                cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)
        stream.synchronize()

    # measured runs
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

    avg_ms = float(np.mean(latencies))
    outputs = {}
    for name, buf in buffers.items():
        if not buf["is_input"]:
            # convert host buffer to numpy array shaped
            arr = np.frombuffer(buf["host"], dtype=np.dtype(buf["dtype"]))
            outputs[name] = arr.reshape(buf["shape"]).astype(np.float32)  # cast to float32 for comparison
    return avg_ms, outputs

def compute_error_metrics(ref, other):
    """ref, other : numpy arrays (float32), same shape
       返回 dict: L1_mean, RMSE, PSNR (dB)
    """
    # ensure shape equal
    if ref.shape != other.shape:
        return {"error": "shape_mismatch", "ref_shape": ref.shape, "other_shape": other.shape}
    diff = ref - other
    abs_diff = np.abs(diff)
    l1 = float(np.mean(abs_diff))
    mse = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse) if mse >= 0 else float("nan")
    # PSNR: use peak = max(abs(ref)) or 1.0 fallback
    peak = float(np.max(np.abs(ref)))
    if peak == 0:
        peak = 1.0
    psnr = 20.0 * math.log10(peak / (math.sqrt(mse) + 1e-12)) if mse > 0 else float("inf")
    return {"L1_mean": l1, "RMSE": rmse, "PSNR_dB": psnr}

def main():
    results = {}
    reference_outputs = None
    reference_name = None

    print(f"[INFO] TensorRT version: {trt.__version__}")

    for idx, path in enumerate(engine_paths):
        base = os.path.basename(path)
        print(f"\n[INFO] Processing engine ({idx+1}/{len(engine_paths)}): {base}")

        engine, err, mem_used_mib, gpu_total_mib = safe_load_engine(path)
        if engine is None:
            print(f"  [ERROR] load failed: {err}")
            results[base] = {"path": path, "error": err}
            continue
        print(f"  [OK] Engine loaded. approx GPU mem used during load: {mem_used_mib:.2f} MiB (GPU total {gpu_total_mib:.1f} MiB)")

        # run inference and measure latency
        try:
            avg_ms, outputs = run_engine(engine, input_shapes, n_runs=N_RUNS)
            fps = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
            print(f"  → Avg latency: {avg_ms:.3f} ms | Throughput: {fps:.2f} FPS")
        except Exception as e:
            results[base] = {"path": path, "error": f"inference failed: {repr(e)}"}
            print(f"  [ERROR] inference failed: {repr(e)}")
            continue

        # save outputs to temporary npy (and include dtype/shape)
        out_info = {}
        for name, arr in outputs.items():
            out_info[name] = {"shape": arr.shape, "dtype": str(arr.dtype)}
        results[base] = {
            "path": path,
            "avg_latency_ms": avg_ms,
            "throughput_fps": fps,
            "mem_used_mib": mem_used_mib,
            "bindings": out_info
        }

        # set first successfully loaded engine as reference
        if reference_outputs is None:
            reference_outputs = outputs
            reference_name = base
            results[base]["role"] = "reference"
            # save reference outputs for possible later checks
            for nm, arr in outputs.items():
                np.save(f"ref_{base}_{nm}.npy", arr)
        else:
            # compute errors against reference where binding names match
            per_binding_errors = {}
            for out_name, arr in outputs.items():
                if out_name in reference_outputs:
                    metrics = compute_error_metrics(reference_outputs[out_name], arr)
                    per_binding_errors[out_name] = metrics
                else:
                    per_binding_errors[out_name] = {"error": "no matching binding in reference"}
            results[base]["errors_vs_reference"] = per_binding_errors

    # save JSON report
    with open("trt_compare_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[INFO] All done. Results saved to trt_compare_report.json")

if __name__ == "__main__":
    main()
