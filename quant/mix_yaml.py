import yaml
import math

def get_min(val):
    if isinstance(val, (list, tuple)):
        return min(val)
    return val

def get_max(val):
    if isinstance(val, (list, tuple)):
        return max(val)
    return val


yaml_path = "/home/extra/share/mix/lightstereo_ptq_out/lightstereo_s_sceneflow_general_opt_256_512_sim_conv_quant_param.yaml"

# 读取 YAML 文件
with open(yaml_path, "r", encoding="utf-8") as f:
    quant_data = yaml.safe_load(f)

layers = quant_data["layers"]

modified_layers = 0

for layer in layers:
    name = layer["layername"]
    if "weight" in name.lower():
        continue  # 忽略带 weight 的层

    layer_min = get_min(layer["min"])
    layer_max = get_max(layer["max"])
    diff = layer_max - layer_min

    if diff > 30:
        nbit = 16
        layer["nbit"] = nbit
        layer["static"] = 1

        # 计算新的 scale 和 zp
        if layer.get("symmetry", False):
            scale = max(abs(layer_min), abs(layer_max)) / (2 ** (nbit - 1) - 1)
            zp = 0
        else:
            scale = (layer_max - layer_min) / (2 ** nbit - 1)
            zp = round(-layer_min / scale)

        layer["scale"] = scale
        layer["zp"] = zp

        modified_layers += 1
        print(f"✅ 修改: {name}, diff={diff:.2f}, scale={scale:.6g}, zp={zp}")

print(f"\n共修改 {modified_layers} 个层 (diff>60)")

# 保存新的 YAML 文件
out_path = yaml_path.replace(".yaml", "_modified.yaml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(quant_data, f, allow_unicode=True, sort_keys=False)

print(f"已保存到: {out_path}")
