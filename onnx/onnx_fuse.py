"""
onnx_graph_fuse.py
- 功能1: 在 ONNX 图中将 BatchNormalization 折叠(fold)进前面的 Conv（如果安全可行）
- 功能2: 检测 Conv -> Relu 模式并示例性重写/标记（注：真正的运行时融合通常由后端执行）


| 序号 | 可融合算子组合                     | 对应优化 pass                                                 | 说明                                     |
| -- | --------------------------- | --------------------------------------------------------- | -------------------------------------- |
| 1  | `Conv + BatchNormalization` | `fuse_bn_into_conv`                                       | 将卷积和 BN 融合，消除额外的 scale/shift 运算，提高推理效率 |
| 2  | `Conv + Relu`               | `fuse_pad_into_conv` + `fuse_consecutive_transposes`      | 将卷积与 ReLU 激活函数融合为 FusedConvRelu        |
| 3  | `Conv + LeakyRelu`          | `fuse_pad_into_conv`                                      | 卷积 + LeakyRelu 激活融合                    |
| 4  | `Conv + Sigmoid`            | `fuse_pad_into_conv`                                      | 卷积 + Sigmoid 激活融合                      |
| 5  | `Gemm/MatMul + Add`         | `fuse_add_bias_into_gemm`                                 | 将全连接层加偏置融合，减少算子数量                      |
| 6  | `Gemm/MatMul + Relu`        | `fuse_add_bias_into_gemm` + `fuse_consecutive_transposes` | 将全连接层与激活函数融合                           |
| 7  | `Conv + Add + Relu`         | `fuse_pad_into_conv` + `fuse_consecutive_transposes`      | 卷积→加偏置→ReLU 融合为 FusedConvAddRelu       |
| 8  | `Add/Sub/Mul/Div/Pow`（常量输入） | `eliminate_constant` / `constant_folding`                 | 对常量进行折叠计算，减少运行时开销                      |
| 9  | `Concat + Reshape`（常量）      | `fuse_consecutive_reshapes`                               | 连续常量 reshape/concat 可提前合并              |
| 10 | `Reshape + Reshape`         | `fuse_consecutive_reshapes`                               | 合并连续 reshape 操作                        |
| 11 | `Slice + Slice`             | `fuse_consecutive_slices`                                 | 合并连续 Slice 节点                          |
| 12 | `Transpose + Transpose`     | `fuse_consecutive_transposes`                             | 合并连续转置操作                               |
| 13 | `Transpose + MatMul/Gemm`   | `fuse_transpose_into_gemm` / `fuse_transpose_into_matmul` | 调整维度，减少无用 transpose                    |
| 14 | `Identity`                  | `eliminate_identity`                                      | 删除冗余 Identity 节点                       |
| 15 | `Dropout`                   | `eliminate_deadend`                                       | 推理阶段可删除 Dropout 节点                     |
| 16 | `Pad + Conv`                | `fuse_pad_into_conv`                                      | 将 padding 融入卷积算子，减少额外 Pad 节点           |
| 17 | `BatchNorm + Mul/Add`       | `fuse_bn_into_conv`                                       | 将 BN 的 scale/shift 融入 Conv 权重，提升性能     |
| 18 | `Conv + Mul(const)`         | `fuse_mul_into_conv`                                      | 将卷积权重按常量 scale 融合                      |
| 19 | `Concat + Conv`             | `fuse_concat_into_conv`                                   | 合并多输入卷积的 Concat 操作（部分优化器支持）            |
| 20 | `Pad + MaxPool/AvgPool`     | `fuse_pad_into_pool`                                      | 将 padding 融入池化层，减少节点                   |



ONNX 图优化的算子融合重点：
卷积类融合：Conv + BN + Activation
全连接类融合：Gemm/MatMul + Add + Activation
常量折叠：Add/Sub/Mul/Div/Pow
转置优化：Transpose 合并或消除
冗余节点删除：Identity、Dropout
Pad/Reshape/Slice 简化
跨节点融合：Concat/Reshape/Conv Bias


1. Conv + BatchNorm 融合

场景：推理时可以将卷积和后续的 BatchNorm 融合为一个卷积，消除额外的加减乘除操作。

典型算子组合：

Conv + BatchNormalization → 融合为 Conv

优化 pass：fuse_bn_into_conv


2. Conv + Activation 融合

场景：卷积后紧跟 ReLU 或 Sigmoid 等激活函数时，可以将它们融合成一个 FusedConv。

典型算子组合：

Conv + Relu → FusedConvRelu

Conv + Sigmoid → FusedConvSigmoid

Conv + LeakyRelu → FusedConvLeakyRelu

优化 pass：fuse_pad_into_conv, fuse_consecutive_transposes（可减少 transpose

3. MatMul / Gemm + Activation

场景：全连接层后紧跟激活函数。

典型算子组合：

Gemm + Relu → FusedGemmRelu

MatMul + Add + Relu → 可融合成 FusedMatMulRelu

优化 pass：fuse_add_bias_into_gemm，fuse_consecutive_transp



4. 常量折叠（Constant Folding）

场景：图中出现常量运算，如 Add(Const, Const)，可以提前计算。

典型算子：

Add, Sub, Mul, Div, Pow

Concat、Reshape 等如果输入为常量

优化 pass：eliminate_constant, constant_folding


5. 转置（Transpose）优化

场景：连续或冗余的 transpose 可以合并或删除。

典型算子组合：

Transpose + Transpose → 可消除或合并

Transpose + MatMul → 可以通过调整维度减少 transpose

优化 pass：

eliminate_nop_transpose

fuse_consecutive_transposes

fuse_transpose_into_gemm


6. Identity / Dropout 删除

场景：推理阶段 Dropout 可以忽略，Identity 节点冗余。

典型算子：

Dropout → 可删除

Identity → 可删除

优化 pass：

eliminate_identity

eliminate_deadend（未使用输出）



7. Pad / Slice / Reshape 优化

场景：连续的 Pad、Reshape、Slice 可以合并或简化。

典型算子组合：

Pad + Conv → 可融合

Reshape + Reshape → 合并

Slice + Slice → 合并

优化 pass：

fuse_pad_into_conv

fuse_consecutive_reshapes

fuse_consecutive_slices



8. 常见其他融合算子
融合类型	算子组合例子
Conv + Add + Relu	Conv → Add(bias) → Relu
Conv + Mul	Conv → Mul(const scale)
BatchNorm + Mul/Add	BN → Mul/Add → Fold into Conv
Gemm + Add	MatMul → Add(bias)
Concat + Reshape	多个常量/权重可以提前合并





| 序号 | 可融合算子组合                     | 对应优化 pass                                                 | 说明                                     |
| -- | --------------------------- | --------------------------------------------------------- | -------------------------------------- |
| 1  | `Conv + BatchNormalization` | `fuse_bn_into_conv`                                       | 将卷积和 BN 融合，消除额外的 scale/shift 运算，提高推理效率 |
| 2  | `Conv + Relu`               | `fuse_pad_into_conv` + `fuse_consecutive_transposes`      | 将卷积与 ReLU 激活函数融合为 FusedConvRelu        |
| 3  | `Conv + LeakyRelu`          | `fuse_pad_into_conv`                                      | 卷积 + LeakyRelu 激活融合                    |
| 4  | `Conv + Sigmoid`            | `fuse_pad_into_conv`                                      | 卷积 + Sigmoid 激活融合                      |
| 5  | `Gemm/MatMul + Add`         | `fuse_add_bias_into_gemm`                                 | 将全连接层加偏置融合，减少算子数量                      |
| 6  | `Gemm/MatMul + Relu`        | `fuse_add_bias_into_gemm` + `fuse_consecutive_transposes` | 将全连接层与激活函数融合                           |
| 7  | `Conv + Add + Relu`         | `fuse_pad_into_conv` + `fuse_consecutive_transposes`      | 卷积→加偏置→ReLU 融合为 FusedConvAddRelu       |
| 8  | `Add/Sub/Mul/Div/Pow`（常量输入） | `eliminate_constant` / `constant_folding`                 | 对常量进行折叠计算，减少运行时开销                      |
| 9  | `Concat + Reshape`（常量）      | `fuse_consecutive_reshapes`                               | 连续常量 reshape/concat 可提前合并              |
| 10 | `Reshape + Reshape`         | `fuse_consecutive_reshapes`                               | 合并连续 reshape 操作                        |
| 11 | `Slice + Slice`             | `fuse_consecutive_slices`                                 | 合并连续 Slice 节点                          |
| 12 | `Transpose + Transpose`     | `fuse_consecutive_transposes`                             | 合并连续转置操作                               |
| 13 | `Transpose + MatMul/Gemm`   | `fuse_transpose_into_gemm` / `fuse_transpose_into_matmul` | 调整维度，减少无用 transpose                    |
| 14 | `Identity`                  | `eliminate_identity`                                      | 删除冗余 Identity 节点                       |
| 15 | `Dropout`                   | `eliminate_deadend`                                       | 推理阶段可删除 Dropout 节点                     |
| 16 | `Pad + Conv`                | `fuse_pad_into_conv`                                      | 将 padding 融入卷积算子，减少额外 Pad 节点           |
| 17 | `BatchNorm + Mul/Add`       | `fuse_bn_into_conv`                                       | 将 BN 的 scale/shift 融入 Conv 权重，提升性能     |
| 18 | `Conv + Mul(const)`         | `fuse_mul_into_conv`                                      | 将卷积权重按常量 scale 融合                      |
| 19 | `Concat + Conv`             | `fuse_concat_into_conv`                                   | 合并多输入卷积的 Concat 操作（部分优化器支持）            |
| 20 | `Pad + MaxPool/AvgPool`     | `fuse_pad_into_pool`                                      | 将 padding 融入池化层，减少节点                   |


"""

import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto, ModelProto, NodeProto, GraphProto
from copy import deepcopy
from typing import Dict, Tuple, Optional, List

# ---------------------
# 工具函数
# ---------------------
def initializer_to_numpy_map(graph: GraphProto) -> Dict[str, np.ndarray]:
    """将 graph.initializer 转为名字->numpy 数组 dict"""
    m = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        m[init.name] = arr
    return m

def find_node_by_output_name(graph: GraphProto, out_name: str) -> Optional[NodeProto]:
    for node in graph.node:
        if out_name in node.output:
            return node
    return None

def consumers_of_tensor(graph: GraphProto, tensor_name: str) -> List[NodeProto]:
    """返回使用给定 tensor_name 的节点列表（输入）"""
    out = []
    for n in graph.node:
        if tensor_name in n.input:
            out.append(n)
    return out

def remove_node(graph: GraphProto, node: NodeProto):
    """从 graph.node 中移除 NodeProto（按对象 identity 移除）"""
    nodes = [n for n in graph.node if n is not node]
    del graph.node[:]
    graph.node.extend(nodes)

def replace_initializer(graph: GraphProto, name: str, np_arr: np.ndarray):
    """替换或添加 initializer（按 name）"""
    # check exists
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].raw_data = numpy_helper.from_array(np_arr, name).raw_data
            graph.initializer[i].data_type = numpy_helper.from_array(np_arr, name).data_type
            graph.initializer[i].dims[:] = list(np_arr.shape)
            return
    # not found -> append
    graph.initializer.extend([numpy_helper.from_array(np_arr, name)])

# ---------------------
# BatchNorm 融合实现
# ---------------------
def fold_batchnorm_into_conv(model: ModelProto) -> Tuple[ModelProto, int]:
    """
    遍历 graph，寻找 Conv -> BatchNormalization 串联并尝试折叠 BN 到 Conv：
      y = Conv(x, W, b)  # b may be absent
      y_bn = scale * (y - mean) / sqrt(var + eps) + B
    融合后，计算新的 W' 和 b'，删除 BatchNormalization 节点及它的 init（如果只被该 BN 使用）。
    返回 (new_model, fused_count)
    """
    graph = model.graph
    name_to_np = initializer_to_numpy_map(graph)

    fused = 0
    # 使用浅拷贝遍历，因为我们可能修改 graph.node
    nodes = list(graph.node)
    for node in nodes:
        if node.op_type != "Conv":
            continue
        conv_node = node
        conv_out = conv_node.output[0]

        # 找到下游仅由 BN 使用的情况
        bn_node = find_node_by_output_name(graph, conv_out)
        # careful: find_node_by_output_name 查找给定输出为 node.output 的节点 —— 但我们需要找以 conv_out 为输入的 BN
        # 更准确：遍历所有节点看第一个输入是否等于 conv_out 并且是 BatchNormalization
        bn_node = None
        for n in graph.node:
            if n.op_type == "BatchNormalization" and conv_out in n.input:
                # ensure BN's input that equals conv_out is the first input (X)
                # BN inputs: [X, scale, B, mean, var]
                # only fuse when BN's X is exactly conv_out
                if n.input[0] == conv_out:
                    bn_node = n
                    break
        if bn_node is None:
            continue

        # ensure bn_node 输出仅被单一消费者使用（否则删除 BN 可能破坏其他节点）
        bn_out = bn_node.output[0]
        consumers = consumers_of_tensor(graph, bn_out)
        if len(consumers) != 1:
            # 如果 BN 的输出被多个节点消费，则不安全跳过
            continue

        # 检查 BN 的 scale/B/mean/var 是否都存在于 initializer（常量）中（必须是 inference 模式）
        try:
            scale_name = bn_node.input[1]
            B_name = bn_node.input[2]
            mean_name = bn_node.input[3]
            var_name = bn_node.input[4]
            scale = name_to_np[scale_name]
            B = name_to_np[B_name]
            mean = name_to_np[mean_name]
            var = name_to_np[var_name]
        except Exception:
            # 其中任一不是 initializer（可能来自其他节点），不合并
            continue

        # read eps 属性（默认是 1e-5），如果存在则采用
        eps = 1e-5
        for attr in bn_node.attribute:
            if attr.name == "epsilon" or attr.name == "eps":
                eps = helper.get_attribute_value(attr)
                break

        # 获取 Conv 的权重和 bias （bias 可能没有）
        W_name = conv_node.input[1]
        W = name_to_np.get(W_name, None)
        if W is None:
            # 没有权重 init，无法合并
            continue
        # bias
        if len(conv_node.input) > 2:
            b_name = conv_node.input[2]
            b = name_to_np.get(b_name, None)
            if b is None:
                b = np.zeros(W.shape[0], dtype=W.dtype)
        else:
            b = np.zeros(W.shape[0], dtype=W.dtype)
            # will need to add bias init later

        # BN scale parameters are per-channel; shape should match output channels
        # compute the multiplier per output channel: scale / sqrt(var + eps)
        denom = np.sqrt(var + eps)
        multiplier = scale / denom  # shape: (C_out,)

        # W shape: (C_out, C_in, kH, kW) for Conv2d
        # multiply W per output channel
        W_new = W * multiplier.reshape((-1, 1, 1, 1))
        # new bias: b' = (b - mean) * multiplier + B
        b_new = (b - mean) * multiplier + B

        # 写回 initializer
        replace_initializer(graph, W_name, W_new.astype(W.dtype))
        # ensure bias exists as initializer and Conv node input contains it
        if len(conv_node.input) > 2:
            # replace existing bias initializer
            replace_initializer(graph, conv_node.input[2], b_new.astype(b.dtype))
        else:
            # create a new bias initializer and append to conv_node.input
            bias_name = conv_node.name + "_bias_after_fold" if conv_node.name else conv_node.output[0] + "_bias"
            # add initializer
            graph.initializer.extend([numpy_helper.from_array(b_new.astype(b.dtype), bias_name)])
            # modify conv node inputs to include bias
            conv_node.input.append(bias_name)

        # 删除 BN 节点：把 conv 的输出名替换为 bn 的输出名（保持后续节点输入一致）
        # 方法：将 conv_node.output[0] 改为 bn_node.output[0]，然后删除 bn_node
        conv_node.output[0] = bn_node.output[0]

        # 删除 bn_node 及其 initializer（如果这些 initializer 不被其他节点使用）
        # 首先删除 bn_node
        remove_node(graph, bn_node)
        # 删除 bn initializers（scale, B, mean, var）如果没有别处消费
        for nm in [scale_name, B_name, mean_name, var_name]:
            # check consumers of nm
            used_elsewhere = False
            for n2 in graph.node:
                if nm in n2.input:
                    used_elsewhere = True
                    break
            if not used_elsewhere:
                # remove initializer from graph.initializer
                graph.initializer[:] = [init for init in graph.initializer if init.name != nm]

        # update name_to_np map for subsequent fusions
        name_to_np = initializer_to_numpy_map(graph)
        fused += 1

    return model, fused

# ---------------------
# Conv + Relu 检测 & 标注示例
# ---------------------
def detect_conv_relu_and_mark(graph: GraphProto) -> int:
    """
    简单检测 Conv->Relu 模式并将 Relu 删除，改为在 Conv node 的 attribute 中添加 annotation 'fused_activation'='Relu'
    注意：这不是标准 ONNX 语义（Conv 本身不执行 Relu），仅用于 graph-level 标注示例。
    真正要把两个操作合并为可执行的“fused kernel”，需要运行时/后端支持。
    """
    marked = 0
    for node in list(graph.node):
        if node.op_type != "Conv":
            continue
        conv_out = node.output[0]
        # find relu node that consumes conv_out
        for n in graph.node:
            if n.op_type == "Relu" and n.input and n.input[0] == conv_out:
                # ensure Relu 的输出只有单个消费者（安全起见）
                relu_out = n.output[0]
                consumers = consumers_of_tensor(graph, relu_out)
                if len(consumers) > 1:
                    continue
                # 标注 conv node（添加 attribute fused_activation=Relu）
                # 这里我们只添加一个属性作为示例（运行时不会识别它，除非你实现 custom kernel）
                node.attribute.extend([helper.make_attribute("fused_activation", "Relu")])
                # 将 conv 的输出名称变成 relu 的输出名称，这样后续节点不需要修改
                node.output[0] = relu_out
                # 删除 relu node
                remove_node(graph, n)
                marked += 1
                break
    return marked

# ---------------------
# 主函数（示例用）
# ---------------------
def main(in_path: str, out_path: str):
    model = onnx.load(in_path)
    print("Loaded model:", in_path)
    print("Graph nodes before:", len(model.graph.node))

    model, fused_bn_count = fold_batchnorm_into_conv(model)
    print("BatchNorm folded count:", fused_bn_count)

    marked_relu = detect_conv_relu_and_mark(model.graph)
    print("Conv->Relu patterns annotated (marked):", marked_relu)

    # 校验并保存
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print("Saved optimized model to:", out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="input onnx model")
    p.add_argument("--output", "-o", default="model_opt.onnx", help="output onnx model")
    args = p.parse_args()
    main(args.input, args.output)
    
    
## 图优化  新算子 tensor
