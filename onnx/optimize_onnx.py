import onnx
import onnxoptimizer
import onnxruntime as ort
import numpy as np
import subprocess

# ========== Step 1: 使用 onnx-simplifier 简化模型 ==========
def simplify_onnx(input_model, simplified_model):
    print(">>> Simplifying ONNX model...")
    subprocess.run(["python3", "-m", "onnxsim", input_model, simplified_model], check=True)
    



# ========== Step 1.5: 修复 initializer 引用 ==========
def fix_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    将所有 initializer 确保加入 graph.input
    避免 onnxoptimizer 报 Unresolved value references
    """
    input_names = [i.name for i in model.graph.input]
    for init in model.graph.initializer:
        if init.name not in input_names:
            # 使用 initializer 的 shape 和 type 创建 TensorValueInfo
            vi = onnx.helper.make_tensor_value_info(
                init.name,
                init.data_type,
                list(init.dims)
            )
            model.graph.input.append(vi)
    return model
    
'''

['adjust_add', 'rename_input_output', 'set_unique_name_for_nodes', 'nop', 'eliminate_nop_cast', 'eliminate_nop_dropout',
'eliminate_nop_flatten', 'extract_constant_to_initializer', 'eliminate_if_with_const_cond', 'eliminate_nop_monotone_argmax', 
'eliminate_nop_pad', 'eliminate_nop_concat', 'eliminate_nop_split', 'eliminate_nop_expand', 'eliminate_shape_gather', 
'eliminate_slice_after_shape', 'eliminate_nop_transpose', 'fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'fuse_consecutive_concats', 

'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes', 'fuse_consecutive_transposes', 
'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv', 'fuse_pad_into_pool', 'fuse_transpose_into_gemm', 'replace_einsum_with_matmul', 
'lift_lexical_references', 'split_init', 'split_predict', 'fuse_concat_into_reshape', 'eliminate_nop_reshape', 'eliminate_nop_with_unit', 
'eliminate_common_subexpression', 'fuse_qkv', 'fuse_consecutive_unsqueezes', 'eliminate_deadend', 'eliminate_identity', 'eliminate_shape_op', 
'fuse_consecutive_slices', 'eliminate_unused_initializer', 'eliminate_duplicate_initializer', 'adjust_slice_and_matmul', 'rewrite_input_dtype']



'''


'''
import onnx
from onnx import helper, TensorProto

# Pad -> MaxPool
pad = helper.make_node(
    "Pad", 
    inputs=["input"], outputs=["pad_out"],
    mode="constant",
    pads=[0, 0, 1, 1, 0, 0, 1, 1],  # only pad H/W
    value=0.0
)

pool = helper.make_node(
    "MaxPool",
    inputs=["pad_out"], outputs=["pool_out"],
    kernel_shape=[3, 3],
    strides=[1, 1],
    pads=[0, 0, 0, 0]
)


✅ 1. split_init —— 拆出静态部分

正如你前面理解的：

识别所有与输入无关的节点；

拆成 Init Graph；

Predict Graph 用 Init 输出替代这些节点。

输出结果：

InitGraph.onnx

PredictGraph.onnx（但此时还包含一些未优化的常量计算）

✅ 2. constant_folding —— 常量折叠

这是接下来非常关键的一步。

把所有可以在编译时计算出来的节点，提前计算成 Constant。

例如：
Add(Constant(2), Constant(3)) → Constant(5)
或：
MatMul(Constant(W), Constant(X)) → Constant(Y)
在 split_init 之后，Init Graph 中往往有大量常量流动路径：

权重常量

shape 计算

BN 参数融合

precompute 位置编码

经过 constant folding，这些节点就会被直接计算并替换为常量数据。
→ 减少运行时计算开销。

✅ 3. eliminate_deadend —— 删除死节点（无用子图）

拆分和常量折叠之后，往往会遗留一些：

未被使用的节点；

断开的子图；

临时张量。

这个 Pass 负责：

从输出端向上追溯依赖；

删除所有不再被引用的节点；

清理掉孤立的计算支路。

💡 类似于编译器里的「Dead Code Elimination」。

✅ 4. split_predict —— 保留推理主图

到这里再运行 split_predict，做两件事：

保留主推理图（Predict Graph）；

移除所有 Init-only 计算和中间缓存；

更新输入签名，使 Predict Graph 仅依赖：

原始模型输入；

Init Graph 输出（作为外部常量）。

最终形成独立可执行的 “轻量推理图”。

✅ 5. （可选）graph_cleanup —— 清理图结构

在拆分和删除操作之后，有时图中还会遗留：

无用 initializer；

重复 name；

空 shape；

不必要的 Cast/Identity。

graph_cleanup 会统一清理这些问题，保证模型结构合法。


'''

# ========== Step 2: 使用 onnxoptimizer 进一步图优化 ==========
def optimize_onnx(input_model, optimized_model):
    print(">>> Optimizing ONNX model with onnxoptimizer...")
    
    passes = onnxoptimizer.get_available_passes()
    print(onnxoptimizer.get_available_passes())

    model = onnx.load(input_model)
    model = fix_initializers(model)  # 修复 initializer 引用

    # 打印优化前的算子统计
    original_nodes = len(model.graph.node)
    original_ops = [node.op_type for node in model.graph.node]
    print(f"优化前: {original_nodes} 个算子")
    print(f"算子分布: { {op: original_ops.count(op) for op in set(original_ops)} }")

    # 可选优化 pass（根据需求增减）
    # passes = [
    #     "eliminate_identity",             # 删除 Identity 节点
    #     "eliminate_nop_transpose",        # 删除无效的 Transpose
    #     "eliminate_nop_pad",              # 删除无效的 Pad
    #     "eliminate_deadend",              # 删除死节点
    #     "fuse_consecutive_transposes",    # 融合连续的 Transpose
    #     "fuse_bn_into_conv",              # 融合 BN 到 Conv
    # ]
    
    
    print(len(passes))
    exclude_passes = [
        "split_init",
        "rewrite_input_dtype", 
        "adjust_add", 
        "adjust_slice_and_matmul",
        "eliminate_nop_cast"
]

    
    
    passes = [e for e in passes if e not in exclude_passes]
    print(len(passes))

    optimized_model_proto = onnxoptimizer.optimize(model, passes)

    # 打印优化后的算子统计
    optimized_nodes = len(optimized_model_proto.graph.node)
    optimized_ops = [node.op_type for node in optimized_model_proto.graph.node]
    print(f"优化后: {optimized_nodes} 个算子")
    print(f"算子分布: { {op: optimized_ops.count(op) for op in set(optimized_ops)} }")

    # 保存优化后的模型
    onnx.save(optimized_model_proto, optimized_model)

# ========== Step 3: 使用 ONNX Runtime 进行推理测试 ==========
def run_inference(model_path):
    print(">>> Running inference with ONNX Runtime...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    session = ort.InferenceSession(model_path, sess_options)

    # 获取输入信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    print(f"Input name: {input_name}, shape: {input_shape}, type: {input_type}")

    # 构造一个随机输入（仅用于测试）
    dummy_input = np.random.randn(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np.float32)

    outputs = session.run(None, {input_name: dummy_input})
    print(f"Output shape: {outputs[0].shape}")

# ========== Main ==========
if __name__ == "__main__":
    original_model = "./output/onnx/lightstereo_s_sceneflow_general_opt_256_512_sim_conv.onnx"
    # original_model = "./output/model_480x640.onnx"
    simplified_model = "./output/onnx/model_simplified.onnx"
    optimized_model = "./output/onnx/model_optimized.onnx"

    simplify_onnx(original_model, simplified_model)
    optimize_onnx(simplified_model, optimized_model)
    # run_inference(optimized_model)

    print(">>> All steps done!")