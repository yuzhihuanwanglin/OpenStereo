import json
from collections import Counter

# 读取 JSON 文件
with open("trt/export_layer_fp32_simple.json", "r", encoding="utf-8") as f:
    data = json.load(f)

Layers = data.get("Layers", [])


def get_layer_type(layer_name):
    """提取层类型，拆分融合层，处理特殊命名"""
    layer_name = layer_name.strip()
    if not layer_name:
        return None
    sublayers = layer_name.split('+')  # 拆分融合层
    types = []
    for sub in sublayers:
        sub = sub.strip()
        if not sub:
            continue
        first = sub.split()[0]  # 拿第一个单词
        first = first.split('_')[0]  # 去掉下划线后编号
        # 特殊命名归类
        if first.startswith('PWN('):
            first = 'PWN'
        elif first.startswith('input'):
            first = 'input'
        elif first.startswith('__myl'):
            first = '__myl'
        elif first.startswith('Reformatting'):
            first = 'Reformatting'
        elif first.startswith('Resize'):
            first = 'Resize'
        elif first.startswith('Softmax'):
            first = 'Softmax'
        elif first.startswith('ReduceSum'):
            first = 'ReduceSum'
        types.append(first)
    return types

def analyze_layers(layers):
    total_layers_after = len(layers)
    fusion_layers = [l for l in layers if '+' in l]
    num_fusion_layers = len(fusion_layers)
    original_layers = total_layers_after + sum(len(l.split('+')) - 1 for l in fusion_layers)

    # 提取各类型层
    all_types = []
    for l in layers:
        types = get_layer_type(l)
        if types:
            all_types.extend(types)
    counter = Counter(all_types)

    # 输出结果
    print(f"当前节点总数 (融合后): {total_layers_after}")
    print(f"融合层数 (包含 '+'): {num_fusion_layers}")
    print(f"估计原始层数 (拆分融合前): {original_layers}")
    print("\n各类型层统计:")
    for k, v in counter.items():
        print(f"{k}: {v}")

# 调用分析
analyze_layers(Layers)