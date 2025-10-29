import onnx
import json
import os

def get_onnx_layers(onnx_path):
    model = onnx.load(onnx_path)
    layers = []
    for node in model.graph.node:
        layers.append({
            "name": node.name if node.name else node.output[0],
            "type": node.op_type
        })
    return layers

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
    n1 = 0  # 1:1 完全匹配
    n2 = 0  # ONNX 被融合消失
    n3 = 0  # 融合后的 TRT 层
    n4 = 0  # TRT 新增层
    n5 = 0  # ONNX 消失层

    onnx_name_set = set([l["name"] for l in onnx_layers])
    onnx_in_trt = set()

    for trt in trt_layers:
        metadata = trt.get("metadata", "")
        if metadata:
            # 解析 Metadata 中的 ONNX 层
            onnx_list = []
            for item in metadata.split("\u001e"):
                item = item.strip()
                if item.startswith("[ONNX Layer:") and item.endswith("]"):
                    name = item[len("[ONNX Layer:"):-1].strip()
                    if name in onnx_name_set:
                        onnx_list.append(name)
                        onnx_in_trt.add(name)
            if len(onnx_list) == 0:
                n4 += 1  # TRT 新增层
            elif len(onnx_list) == 1:
                n1 += 1  # 完全匹配
            else:
                n3 += 1  # 融合后的 TRT 层
                n2 += len(onnx_list)  # ONNX 被融合消失
        else:
            n4 += 1  # TRT 新增层

    # ONNX 消失层
    n5 = len(onnx_name_set - onnx_in_trt)

    stats = {
        "num_onnx": len(onnx_layers),
        "num_trt": len(trt_layers),
        "n1_exact": n1,
        "n2_fused_onnx": n2,
        "n3_fused_trt": n3,
        "n4_trt_added": n4,
        "n5_onnx_disappeared": n5
    }
    return stats

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
    
     # 校验 TRT 层
    lhs_trt = stats['n1_exact'] + stats['n3_fused_trt'] + stats['n4_trt_added']
    trt_check = "✅" if lhs_trt == stats['num_trt'] else "❌"
    print(f"TRT 校验: n1 + n3 + n4 = {lhs_trt}  vs  TRT层数={stats['num_trt']} {trt_check}")

    # 校验 ONNX 层
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

    # 保存结果
    with open("onnx_trt_fine_grained_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("✅ 结果已保存到 onnx_trt_fine_grained_stats.json")
