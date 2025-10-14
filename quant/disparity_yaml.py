import yaml



def get_min(val):
    """返回 val 的最小值，val 可以是 float 或 list/tuple"""
    if isinstance(val, (list, tuple)):
        return min(val)
    else:
        return val

def get_max(val):
    """返回 val 的最大值，val 可以是 float 或 list/tuple"""
    if isinstance(val, (list, tuple)):
        return max(val)
    else:
        return val

# 读取 YAML 文件
with open("/home/extra/share/mgz/lightstereo_ptq_out/lightstereo_s_sceneflow_general_opt_256_512_sim_conv_quant_param.yaml", "r", encoding="utf-8") as f:
    quant_data  = yaml.safe_load(f)

# 访问顶层信息
print(f"版本号: {quant_data['quanttool_version']}")
print(f"时间: {quant_data['datetime']}")
print(f"文件名: {quant_data['file_name']}")

# 访问 layer 列表
layers = quant_data["layers"]
print(f"\n共 {len(layers)} 层量化信息\n")


layer_diffs = []
# 遍历层
for i, layer in enumerate(layers, start=1):
    
    layer_min = get_min(layer['min'])
    layer_max = get_max(layer['max'])
    
    diff = layer_max - layer_min
    
    layer_diffs.append({
        "index": i,
        "layername": layer['layername'],
        "min": layer_min,
        "max": layer_max,
        "diff": diff
    })
    
    print(f"{i:4d} Layer: {layer['layername']}, max: {layer_max}, min: {layer_min} diff:{diff}")
    # print(f"{i:4d} Layer: {layer['layername']}, Scale: {layer['scale']}, zp: {layer['zp']}")
    
sorted_by_diff = sorted(layer_diffs, key=lambda x: x['diff'], reverse=True)
print("\n按 diff 降序排序输出：")
for info in sorted_by_diff[:300]:
    print(f"{info['index']:4d} Layer: {info['layername']}, max: {info['max']}, min: {info['min']}, diff: {info['diff']}")


