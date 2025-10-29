import onnx
import json
import os
from collections import defaultdict, deque

def get_onnx_layers(onnx_path):
    """
    以深度优先（DFS）方式遍历 ONNX 图，生成节点列表。
    每个节点包含 name, type, inputs, outputs。
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # 建立 name -> node 映射
    node_map = {}
    for node in graph.node:
        node_map[node.output[0]] = node

    # 建立依赖图：output_name -> input_names
    input_to_output = defaultdict(list)
    output_to_input = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            output_to_input[inp].append(node.output[0])
            input_to_output[node.output[0]].append(inp)

    # 找到图的输入节点（没有输入来自别的 node）
    graph_inputs = {i.name for i in graph.input}
    graph_outputs = {o.name for o in graph.output}

    # 用于保存顺序
    visited = set()
    ordered_nodes = []

    def dfs(node_output):
        """以输出tensor为索引递归深度遍历"""
        if node_output in visited or node_output not in node_map:
            return
        visited.add(node_output)
        node = node_map[node_output]
        # 先遍历输入节点
        for inp in node.input:
            dfs(inp)
        # 再记录自己
        ordered_nodes.append({
            "name": node.name if node.name else node.output[0],
            "type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output)
        })

    # 从图的输出开始深度遍历
    for out in graph_outputs:
        dfs(out)

    return ordered_nodes


def get_trt_layers(trt_json_path):
    with open(trt_json_path, "r") as f:
        data = json.load(f)
    layers = []
    for l in data["Layers"]:
        layers.append({
            "name": l.get("Name", ""),
            "type": l.get("LayerType", ""),
            "metadata": l.get("Metadata", "")
        })
    return layers


def compute_fine_grained_stats(onnx_layers, trt_layers):
    n1 = n2 = n3 = n4 = n5 = 0
    onnx_name_set = set([l["name"] for l in onnx_layers])
    onnx_in_trt = set()

    for trt in trt_layers:
        metadata = trt.get("metadata", "")
        if metadata:
            onnx_list = []
            for item in metadata.split("\u001e"):
                item = item.strip()
                if item.startswith("[ONNX Layer:") and item.endswith("]"):
                    name = item[len("[ONNX Layer:"):-1].strip()
                    if name in onnx_name_set:
                        onnx_list.append(name)
                        onnx_in_trt.add(name)
            if len(onnx_list) == 0:
                n4 += 1
            elif len(onnx_list) == 1:
                n1 += 1
            else:
                n3 += 1
                n2 += len(onnx_list)
        else:
            n4 += 1

    n5 = len(onnx_name_set - onnx_in_trt)

    return {
        "num_onnx": len(onnx_layers),
        "num_trt": len(trt_layers),
        "n1_exact": n1,
        "n2_fused_onnx": n2,
        "n3_fused_trt": n3,
        "n4_trt_added": n4,
        "n5_onnx_disappeared": n5
    }


def print_stats(stats):
    print("\n=============== 精细层级统计 ================")
    print(f"ONNX 层数 (N_onnx) : {stats['num_onnx']}")
    print(f"TRT 层数  (N_trt)  : {stats['num_trt']}")
    print("=============================================\n")
    print(f"完全匹配层      n1  : {stats['n1_exact']}")
    print(f"ONNX被融合消失  n2  : {stats['n2_fused_onnx']}")
    print(f"融合后的TRT层   n3  : {stats['n3_fused_trt']}")
    print(f"新增层TRT层     n4  : {stats['n4_trt_added']}")
    print(f"ONNX 消失层     n5  : {stats['n5_onnx_disappeared']}")
    print("=============================================\n")

    lhs_trt = stats['n1_exact'] + stats['n3_fused_trt'] + stats['n4_trt_added']
    trt_check = "✅" if lhs_trt == stats['num_trt'] else "❌"
    print(f"TRT 校验: n1 + n3 + n4 = {lhs_trt}  vs  TRT层数={stats['num_trt']} {trt_check}")

    lhs_onnx = stats['n1_exact'] + stats['n2_fused_onnx'] + stats['n5_onnx_disappeared']
    onnx_check = "✅" if lhs_onnx == stats['num_onnx'] else "❌"
    print(f"ONNX 校验: n1 + n2 + n5 = {lhs_onnx}  vs  ONNX层数={stats['num_onnx']} {onnx_check}")


if __name__ == "__main__":
    onnx_path = "onnx/models/da.onnx"
    trt_json_path = "trt/export_layer_fp32.json"

    onnx_layers = get_onnx_layers(onnx_path)
    trt_layers = get_trt_layers(trt_json_path)
    stats = compute_fine_grained_stats(onnx_layers, trt_layers)
    print_stats(stats)

    # 保存统计结果
    # with open("onnx_trt_fine_grained_stats.json", "w", encoding="utf-8") as f:
    #     json.dump(stats, f, indent=2, ensure_ascii=False)
    # print("✅ 统计结果已保存到 onnx_trt_fine_grained_stats.json")

    # 保存 ONNX 层结构（深度优先）
    with open("trt/onnx_layers_dfs.json", "w", encoding="utf-8") as f:
        json.dump(onnx_layers, f, indent=2, ensure_ascii=False)
    print("✅ ONNX 层结构（深度优先）已保存到 onnx_layers_dfs.json")
