import yaml

# ====== 配置 ======
yaml_path = "/home/extra/share/mgz/lightstereo_s_sceneflow_general_opt_256_512_sim_conv_quant_param_simplified.yaml"    # ← 你的量化参数路径
# 风险判断阈值（针对 INT8 非对称量化）
SCALE_HIGH = 0.3
SCALE_MED = 0.1
RANGE_HIGH = 128
RANGE_MED = 60
ZP_MIN = 0
ZP_MAX = 255
ZP_CENTER = 128
ZP_OFFSET_WARN = 100

# ===== 分析函数 =====
def check_layer(layer):
    name = layer.get("layername")
    minv = float(layer.get("min", 0))
    maxv = float(layer.get("max", 0))
    scale = float(layer.get("scale", 0))
    zp = int(layer.get("zp", 0))

    dynamic_range = maxv - minv
    risk = "✅ 正常"
    reason = []

    # 动态范围
    if dynamic_range > RANGE_HIGH:
        risk = "❌ 高风险"
        reason.append(f"动态范围过大({dynamic_range:.1f})")
    elif dynamic_range > RANGE_MED:
        risk = "⚠ 中风险"
        reason.append(f"动态范围较大({dynamic_range:.1f})")

    # scale 检查
    if scale > SCALE_HIGH:
        risk = "❌ 高风险"
        reason.append(f"scale过大({scale:.3f})")
    elif scale > SCALE_MED and risk == "✅ 正常":
        risk = "⚠ 中风险"
        reason.append(f"scale偏大({scale:.3f})")

    # zp 检查
    if zp < ZP_MIN or zp > ZP_MAX:
        risk = "❌ 高风险"
        reason.append(f"zp超出范围({zp})")
    elif abs(zp - ZP_CENTER) > ZP_OFFSET_WARN:
        if risk == "✅ 正常":
            risk = "⚠ 中风险"
        reason.append(f"zp偏离中心({zp})")

    # min<0 检查（非对称情况下有代表性）
    if minv < 0 and risk == "✅ 正常":
        risk = "⚠ 中风险"
        reason.append(f"存在负值({minv:.2f})，依赖非对称偏移")

    return {
        "layer": name,
        "min": minv,
        "max": maxv,
        "scale": scale,
        "zp": zp,
        "dynamic_range": dynamic_range,
        "risk": risk,
        "reason": ", ".join(reason) if reason else "-"
    }

# ===== 主程序 =====
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

layers = data.get("layers", data)
results = [check_layer(layer) for layer in layers]

# ===== 输出结果 =====
print(f"{'Layer':30s} {'Scale':>8s} {'Range':>10s} {'ZP':>6s} {'Risk':>10s}  Reason")
print("-" * 90)
for r in results:
    print(f"{r['layer'][:30]:30s} {r['scale']:8.3f} {r['dynamic_range']:10.2f} {r['zp']:6d} {r['risk']:>10s}  {r['reason']}")

# ===== 汇总 =====
high = [r for r in results if "❌" in r['risk']]
mid = [r for r in results if "⚠" in r['risk']]
print("\n=== 汇总 ===")
print(f"高风险层数: {len(high)}")
print(f"中风险层数: {len(mid)}")
print(f"正常层数: {len(results) - len(high) - len(mid)}")

if high:
    print("\n高风险层列表:")
    for r in high:
        print(f" - {r['layer']} ({r['reason']})")