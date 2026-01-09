"""分析训练日志，提取关键指标"""
import re

log_path = r".\outputs\FSMol_ADKTModel_gnn+ecfp+fc_2025-12-27_01-07-06\train.log"

with open(log_path, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("=" * 70)
print("📊 训练日志分析报告")
print("=" * 70)

# 1. 模型参数
print("\n【1. 模型信息】")
for line in lines:
    if "Num parameters" in line:
        print(f"   {line.strip()}")
    if "modality_gate" in line and "ModalityGate" in line:
        print(f"   ✅ 检测到门控模块: modality_gate")

# 2. 提取训练步骤的loss
print("\n【2. 训练Loss曲线】")
step_losses = []
for line in lines:
    # 匹配类似 "Step 0010 || Mean metric" 的行
    match = re.search(r'Step (\d+).*loss[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
    if match:
        step = int(match.group(1))
        loss = float(match.group(2))
        step_losses.append((step, loss))

if step_losses:
    print(f"   记录了 {len(step_losses)} 个训练步骤的 loss")
    for step, loss in step_losses[:5]:
        print(f"   Step {step:4d}: loss = {loss:.5f}")
    if len(step_losses) > 5:
        print(f"   ...")
        for step, loss in step_losses[-3:]:
            print(f"   Step {step:4d}: loss = {loss:.5f}")
    
    # 计算loss趋势
    if len(step_losses) >= 2:
        first_loss = step_losses[0][1]
        last_loss = step_losses[-1][1]
        trend = "📉 下降" if last_loss < first_loss else "📈 上升"
        print(f"\n   Loss 趋势: {first_loss:.5f} → {last_loss:.5f} ({trend})")
else:
    print("   未找到loss记录")

# 3. 提取验证结果
print("\n【3. 验证结果】")
validations = []
for line in lines:
    # 匹配 "Validated at train step [50/10000], Valid Avg. Prec.: 0.711"
    match = re.search(r'Validated at train step \[(\d+)/(\d+)\].*Prec[.:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
    if match:
        step = int(match.group(1))
        total = int(match.group(2))
        prec = float(match.group(3))
        validations.append((step, prec))

if validations:
    print(f"   共进行了 {len(validations)} 次验证")
    for step, prec in validations:
        print(f"   Step {step:4d}: Valid Avg. Prec. = {prec:.4f}")
    
    best_step, best_prec = max(validations, key=lambda x: x[1])
    print(f"\n   🏆 最佳验证结果: Step {best_step}, Avg. Prec. = {best_prec:.4f}")
else:
    print("   未找到验证记录")

# 4. 检查门控权重输出
print("\n【4. 门控权重记录】")
gate_logs = [line for line in lines if "Gate Weights" in line or "gate" in line.lower()]
gate_weight_logs = [line for line in lines if "Gate Weights:" in line]
if gate_weight_logs:
    print(f"   找到 {len(gate_weight_logs)} 条门控权重记录")
    for log in gate_weight_logs[:5]:
        print(f"   {log.strip()}")
else:
    print("   ⚠️ 未找到门控权重日志（这是因为日志代码是在训练后才添加的）")

# 5. 检查是否有NaN或错误
print("\n【5. 错误检查】")
errors = [line for line in lines if "nan" in line.lower() or "error" in line.lower() or "fail" in line.lower()]
nan_count = len([l for l in errors if "nan" in l.lower()])
if nan_count > 0:
    print(f"   ❌ 发现 {nan_count} 条 NaN 相关记录")
else:
    print("   ✅ 无 NaN 错误")

# 6. 训练总结
print("\n【6. 模型保存记录】")
for line in lines:
    if "Updated" in line and "best" in line:
        print(f"   {line.strip()}")

print("\n" + "=" * 70)
print("📋 分析完成")
print("=" * 70)
