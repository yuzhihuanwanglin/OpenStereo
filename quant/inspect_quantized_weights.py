import onnx
from onnx import numpy_helper

# def print_full_onnx_info(model_path):
#     model = onnx.load(model_path)
#     graph = model.graph

#     print("==== Model Info ====")
#     print(f"IR version: {model.ir_version}")
#     print(f"Producer name: {model.producer_name}")
#     print(f"Producer version: {model.producer_version}")
#     print(f"Domain: {model.domain}")
#     print(f"Model version: {model.model_version}")
#     print("====================\n")

#     # print("==== Inputs ====")
#     # for inp in graph.input:
#     #     dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
#     #     print(f"{inp.name}: {dims}")

#     # print("\n==== Outputs ====")
#     # for out in graph.output:
#     #     dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
#     #     print(f"{out.name}: {dims}")

#     # print("\n==== Nodes ====")
#     # for i, node in enumerate(graph.node):
#     #     print(f"[{i}] {node.op_type}")
#     #     print(f"  Inputs : {list(node.input)}")
#     #     print(f"  Outputs: {list(node.output)}")

#     print("\n==== Initializers ====")
#     for init in graph.initializer:
#         arr = numpy_helper.to_array(init)
#         print(f"{init.name}: shape={arr.shape}, dtype={arr.dtype}")

# # 使用
# print_full_onnx_info("/home/extra/share/mix/lightstereo_s_sceneflow_general_opt_256_512_sim_conv.onnx")

import onnx
import numpy as np
from onnx import numpy_helper

def print_full_onnx_info(model_path):
    # ===== 载入模型 =====
    model = onnx.load(model_path)
    graph = model.graph

    print("=" * 120)
    print(f"📘 ONNX Model: {model_path}")
    print("=" * 120)

    # ===== 模型基本信息 =====
    print("🧩 [Model Info]")
    print(f"IR version       : {model.ir_version}")
    print(f"Producer name    : {model.producer_name}")
    print(f"Producer version : {model.producer_version}")
    print(f"Domain           : {model.domain}")
    print(f"Model version    : {model.model_version}")
    print(f"Doc string       : {model.doc_string}\n")

    # ===== 输入信息 =====
    print("📥 [Model Inputs]")
    for i, inp in enumerate(graph.input):
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  [{i:03d}] {inp.name:<40} shape={shape} dtype={dtype}")
    print("")

    # ===== 输出信息 =====
    print("📤 [Model Outputs]")
    for i, out in enumerate(graph.output):
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"  [{i:03d}] {out.name:<40} shape={shape} dtype={dtype}")
    print("")

    # ===== 节点信息 =====
    print("🧠 [Graph Nodes]")
    for i, node in enumerate(graph.node):
        print(f"[{i:03d}] OpType: {node.op_type:<20}  Name: {node.name}")
        print(f"     Inputs : {list(node.input)}")
        print(f"     Outputs: {list(node.output)}\n")

    # ===== 常量（initializer）信息 =====
    print("🎛️ [Initializers / Weights]")
    for i, init in enumerate(graph.initializer):
        arr = numpy_helper.to_array(init)
        vmin, vmax = float(arr.min()), float(arr.max())
        mean, std = float(arr.mean()), float(arr.std())
        print(f"[{i:03d}] {init.name}")
        print(f"     Shape: {arr.shape}, Dtype: {arr.dtype}")
        print(f"     Min  : {vmin:.6f}, Max: {vmax:.6f}, Mean: {mean:.6f}, Std: {std:.6f}")

        # 如果是量化参数
        lname = init.name.lower()
        if "scale" in lname or "zero_point" in lname:
            print(f"     ⚙️ Quant Param detected: {init.name}")
            flat = arr.flatten()
            print(f"     Values (first 8): {flat[:8]}")
        print("")

    # ===== 汇总统计 =====
    print("📊 [Summary]")
    print(f"Total inputs     : {len(graph.input)}")
    print(f"Total outputs    : {len(graph.output)}")
    print(f"Total nodes      : {len(graph.node)}")
    print(f"Total initializers (weights/bias/scales) : {len(graph.initializer)}")
    print("=" * 120)


# ===== 使用示例 =====
if __name__ == "__main__":
    print_full_onnx_info("/home/extra/share/mix/lightstereo_s_sceneflow_general_opt_256_512_sim_conv.onnx")


