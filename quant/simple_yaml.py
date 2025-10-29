import yaml

yaml_path = "/home/extra/share/symc/lightstereo_ptq_out/lightstereo_s_sceneflow_general_opt_256_512_sim_conv_quant_param.yaml"

# 读取 YAML 文件
with open(yaml_path, "r", encoding="utf-8") as f:
    quant_data = yaml.safe_load(f)

layers = quant_data["layers"]
simplified_layers = []


count = 0
w_count = 0
# 仅保留 min、max、scale、zp 字段
for layer in layers:
    
    if "weight" in layer.get("layername").lower():
        w_count += 1
        continue
    new_layer = {
        "layername": layer["layername"],
    }

    # 如果存在这些字段则保留
    for key in ["min", "max", "scale", "zp"]:
        if key in layer:
            new_layer[key] = layer[key]

    simplified_layers.append(new_layer)
    count+=1

# 保存新的 YAML 文件
quant_data["layers"] = simplified_layers
out_path = yaml_path.replace(".yaml", "_simplified.yaml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(quant_data, f, allow_unicode=True, sort_keys=False)

print(f"✅ 已生成简化版本，仅保留 min/max/scale/zp 字段")
print(f"已保存到: {out_path} with weight:{w_count}  active:{count}")
