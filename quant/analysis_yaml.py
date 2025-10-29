import yaml

# ==============================
# 配置区（可根据实际模型微调）
# ==============================
SCALE_MED = 0.1
SCALE_HIGH = 0.3
RANGE_MED = 60
RANGE_HIGH = 128
ASYM_THRESHOLD = 0.3       # 对称量化下，min/max不平衡超过30%警告
ZP_CENTER_U8 = 128
ZP_CENTER_I8 = 0
ZP_WARN_OFFSET = 100

# ==============================
# 核心分析函数
# ==============================
def analyze_layer(layer, dtype="int8", symmetry=True):
    name = layer.get("layername")
    minv = float(layer.get("min", 0))
    maxv = float(layer.get("max", 0))
    scale = float(layer.get("scale", 0))
    zp = int(layer.get("zp", 0))
    dynamic_range = maxv - minv

    risk = "✅ 正常"
    reason = []
    rec = []  # 建议项

    # 量化误差估计（±scale/2）
    est_error = scale / 2

    # === 通用范围检查 ===
    if dynamic_range > RANGE_HIGH:
        risk = "❌ 高风险"
        reason.append(f"动态范围过大({dynamic_range:.1f})")
        rec.append("考虑clip到较小范围或使用FP16")
    elif dynamic_range > RANGE_MED:
        if risk == "✅ 正常": risk = "⚠ 中风险"
        reason.append(f"动态范围较大({dynamic_range:.1f})")
        rec.append("可尝试轻微clip或重新校准统计")

    # === scale 检查 ===
    if scale > SCALE_HIGH:
        risk = "❌ 高风险"
        reason.append(f"scale过大({scale:.3f})")
        rec.append("量化分辨率过粗，建议重新计算量化区间")
    elif scale > SCALE_MED and risk == "✅ 正常":
        risk = "⚠ 中风险"
        reason.append(f"scale偏大({scale:.3f})")
        rec.append("可考虑缩小动态范围以提高精度")

    # === 对称 / 非对称特性检查 ===
    if symmetry:
        # 对称量化 → 检查 min/max 平衡性
        if abs(abs(minv) - abs(maxv)) / (abs(maxv) + 1e-6) > ASYM_THRESHOLD:
            if risk == "✅ 正常": risk = "⚠ 中风险"
            reason.append("对称量化但min/max偏差较大")
            rec.append("可能需要改为非对称量化")
    else:
        # 非对称量化 → 检查 zp 偏移是否合理
        center = ZP_CENTER_U8 if "u" in dtype.lower() else ZP_CENTER_I8
        if abs(zp - center) > ZP_WARN_OFFSET:
            if risk == "✅ 正常": risk = "⚠ 中风险"
            reason.append(f"zp偏离中心({zp})")
            rec.append("非对称零点偏移较大，可能影响量化精度")

    # === INT8/UINT8 范围检查 ===
    if "u" in dtype.lower():  # UINT8
        if not (0 <= zp <= 255):
            risk = "❌ 高风险"
            reason.append(f"zp超出[0,255]({zp})")
            rec.append("检查量化参数生成逻辑")
    else:  # INT8
        if not (-128 <= zp <= 127):
            risk = "❌ 高风险"
            reason.append(f"zp超出[-128,127]({zp})")
            rec.append("检查量化参数生成逻辑")

    # === 负值检测 ===
    if minv < 0 and "u" in dtype.lower() and symmetry:
        if risk == "✅ 正常": risk = "⚠ 中风险"
        reason.append("UINT8对称量化含负值")
        rec.append("考虑改为INT8或非对称量化")

    return {
        "layer": name,
        "dtype": dtype,
        "symmetry": symmetry,
        "min": minv,
        "max": maxv,
        "scale": scale,
        "zp": zp,
        "dynamic_range": dynamic_range,
        "risk": risk,
        "est_error": est_error,
        "reason": ", ".join(reason) if reason else "-",
        "recommendation": ", ".join(rec) if rec else "无"
    }

# ==============================
# 主程序入口
# ==============================
def analyze_yaml(yaml_path, dtype="int8", symmetry=True):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    layers = data.get("layers", data)
    results = [analyze_layer(layer, dtype, symmetry) for layer in layers]

    # 输出表格
    print(f"\n=== 量化分析报告 ({dtype.upper()} | symmetry={symmetry}) ===\n")
    print(f"{'Layer':30s} {'Scale':>8s} {'Range':>10s} {'ZP':>6s} {'Risk':>10s}  {'Error':>8s}  Reason / 建议")
    print("-" * 120)
    for r in results:
        print(f"{r['layer'][:30]:30s} {r['scale']:8.3f} {r['dynamic_range']:10.2f} {r['zp']:6d} "
              f"{r['risk']:>10s}  ±{r['est_error']:.4f}  {r['reason']} | {r['recommendation']}")

    # 汇总
    high = [r for r in results if "❌" in r['risk']]
    mid = [r for r in results if "⚠" in r['risk']]
    print("\n=== 汇总统计 ===")
    print(f"高风险层数: {len(high)}")
    print(f"中风险层数: {len(mid)}")
    print(f"正常层数: {len(results) - len(high) - len(mid)}")

    if high:
        print("\n高风险层列表:")
        for r in high:
            print(f" - {r['layer']} ({r['reason']})")

# ==============================
# 示例调用
# ==============================
if __name__ == "__main__":
    yaml_path = "/home/extra/share/symc/lightstereo_ptq_out/lightstereo_s_sceneflow_general_opt_256_512_sim_conv_quant_param_simplified.yaml"  # 你的量化文件路径

    # 分别分析四种模式
    analyze_yaml(yaml_path, dtype="int8", symmetry=True)    # INT8 对称
    # analyze_yaml(yaml_path, dtype="int8", symmetry=False)   # INT8 非对称
    # analyze_yaml(yaml_path, dtype="uint8", symmetry=True)   # UINT8 对称
    # analyze_yaml(yaml_path, dtype="uint8", symmetry=False)  # UINT8 非对称
