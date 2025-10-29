import yaml

# ====== 配置 ======
yaml_path = "/home/extra/share/symc/lightstereo_ptq_out/lightstereo_s_sceneflow_general_opt_256_512_sim_conv_quant_param_simplified.yaml"   # ← 你的量化参数路径
SCALE_HIGH = 0.3      # scale > 0.3 表示量化步距大，分辨率粗
SCALE_MED = 0.1       # scale > 0.1 表示轻微风险
RANGE_HIGH = 60       # 动态范围 > 60 视为高风险
RANGE_MED = 30        # 动态范围 > 30 视为中风险
ASYM_THRESHOLD = 0.2  # |min| 和 max 不对称程度（仅用于诊断偏移）

# ====== 分析函数 ======
def check_layer(layer):
    name = layer.get("layername")
    minv = float(layer.get("min", 0))
    maxv = float(layer.get("max", 0))
    scale = float(layer.get("scale", 0))
    zp = int(layer.get("zp", 0))

    dynamic_range = maxv - minv
    # 对称量化理论上 min ≈ -max，这里检查偏差比例
    asym_ratio = abs(abs(minv) - abs(maxv)) / (abs(maxv) + 1e-6)

    risk = "✅ 正常"
    reason = []

    # 1. 动态范围过大
    if dynamic_range > RANGE_HIGH:
        risk = "❌ 高风险"
        reason.append(f"动态范围过大({dynamic_range:.1f})")
    elif dynamic_range > RANGE_MED:
        risk = "⚠ 中风险"
        reason.append(f"动态范围偏大({dynamic_range:.1f})")

    # 2. scale 过大
    if scale > SCALE_HIGH:
        risk = "❌ 高风险"
        reason.append(f"scale过大({scale:.3f})")
    elif scale > SCALE_MED and risk == "✅ 正常":
        risk = "⚠ 中风险"
        reason.append(f"scale偏大({scale:.3f})")

    # 3. 非对称度偏移
    if asym_ratio > ASYM_THRESHOLD:
        if risk == "✅ 正常":
            risk = "⚠ 中风险"
        reason.append(f"非对称偏差({asym_ratio*100:.1f}%)")

    return {
        "layer": name,
        "min": minv,
        "max": maxv,
        "scale": scale,
        "range": dynamic_range,
        "risk": risk,
        "reason": ", ".join(reason) if reason else "-"
    }

# ====== 主流程 ======
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

layers = data.get("layers", [])
results = [check_layer(l) for l in layers]

# ====== 打印结果 ======
print(f"{'Layer':30s} {'Scale':>8s} {'Range':>10s} {'Risk':>10s}  Reason")
print("-" * 80)
for r in results:
    print(f"{r['layer'][:30]:30s} {r['scale']:8.3f} {r['range']:10.2f} {r['risk']:>10s}  {r['reason']}")

# ====== 汇总 ======
high = [r for r in results if "❌" in r["risk"]]
mid = [r for r in results if "⚠" in r["risk"]]
print("\n=== 汇总结果 ===")
print(f"高风险层数: {len(high)}")
print(f"中风险层数: {len(mid)}")
print(f"正常层数: {len(results) - len(high) - len(mid)}")

if high:
    print("\n高风险层列表:")
    for r in high:
        print(f" - {r['layer']} ({r['reason']})")