import json

def load_layer_names(json_path):
    """
    读取 TensorRT 导出的 JSON 文件，返回每层的名称列表。
    兼容两种格式：
    1. simple.json: "Layers" 是字符串列表
    2. detailed.json: "Layers" 是字典列表，每个 dict 有 "Name"
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "Layers" in data:
        layers = data["Layers"]
    else:
        raise ValueError(f"无法识别 JSON 文件: {json_path}")

    layer_names = []
    for layer in layers:
        if isinstance(layer, str):
            # simple.json 的情况
            layer_names.append(layer)
        elif isinstance(layer, dict) and "Name" in layer:
            # detailed.json 的情况
            layer_names.append(layer["Name"])
        else:
            # 遇到无法识别的层，跳过
            continue
    return layer_names

# 读取两份 JSON
default_layers = load_layer_names("trt/export_layer_fp32_simple.json")
detailed_layers = load_layer_names("trt/export_layer_fp32.json")

# 转集合对比
set_default = set(default_layers)
set_detailed = set(detailed_layers)

missing_in_detailed = set_default - set_detailed
extra_in_detailed = set_detailed - set_default

print(f"Default 层数: {len(default_layers)}")
print(f"Detailed 层数: {len(detailed_layers)}\n")

print(f"Default 中但 Detailed 中缺失的层 ({len(missing_in_detailed)}):")
for l in missing_in_detailed:
    print("  ", l)

print(f"\nDetailed 中新增的层 ({len(extra_in_detailed)}):")
for l in extra_in_detailed:
    print("  ", l)
