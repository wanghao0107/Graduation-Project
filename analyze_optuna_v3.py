import sqlite3
import json
import os

# 数据库路径
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optuna.db')

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# LSSeg baseline (study_id 6-10)
print("=" * 70)
print("LSSeg Baseline (test_RITE 02-14 08_30)")
print("=" * 70)

lsseg_study_ids = [6, 7, 8, 9, 10]
lsseg_best_params = []

for study_id in lsseg_study_ids:
    cursor.execute('''
        SELECT t.trial_id, t.number, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ORDER BY tv.value DESC
        LIMIT 1
    ''', (study_id,))
    result = cursor.fetchone()
    
    if result:
        trial_id, number, value = result
        cursor.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id = ?', (trial_id,))
        params = {}
        for pn, pv in cursor.fetchall():
            try:
                params[pn] = json.loads(pv)
            except:
                params[pn] = pv
        lsseg_best_params.append((f"fold_{study_id-6}", value, params))

print("\n各折最佳结果:")
for fold, value, params in lsseg_best_params:
    print(f"  {fold}: Dice={value:.4f}")
    print(f"    lr={params.get('lr', 'N/A'):.6f}, batch_size={params.get('batch_size', 'N/A')}")
    print(f"    weight_decay={params.get('weight_decay', 'N/A'):.6f}")

# 参数统计
print("\n参数统计:")
param_keys = ['lr', 'batch_size', 'weight_decay', 'lr_factor', 'step_size']
for key in param_keys:
    values = [p[2].get(key) for p in lsseg_best_params if key in p[2]]
    if values:
        if isinstance(values[0], (int, float)):
            avg = sum(values) / len(values)
            print(f"  {key}: 平均={avg:.6f}, 范围=[{min(values):.6f}, {max(values):.6f}]")
        else:
            from collections import Counter
            counter = Counter(values)
            print(f"  {key}: 分布={dict(counter)}")

# LSSegSAMLoRA 端到端
print("\n" + "=" * 70)
print("LSSegSAMLoRA 端到端 (03-09 16_23)")
print("=" * 70)

e2e_study_ids = [182, 183, 184, 185, 186]
e2e_best_params = []

for study_id in e2e_study_ids:
    cursor.execute('''
        SELECT t.trial_id, t.number, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ORDER BY tv.value DESC
        LIMIT 1
    ''', (study_id,))
    result = cursor.fetchone()
    
    if result:
        trial_id, number, value = result
        cursor.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id = ?', (trial_id,))
        params = {}
        for pn, pv in cursor.fetchall():
            try:
                params[pn] = json.loads(pv)
            except:
                params[pn] = pv
        e2e_best_params.append((f"fold_{study_id-182}", value, params))

print("\n各折最佳结果:")
for fold, value, params in e2e_best_params:
    print(f"  {fold}: Dice={value:.4f}")
    print(f"    lr={params.get('lr', 'N/A'):.6f}, lora_r={params.get('lora_r', 'N/A')}, prompt_bias={params.get('prompt_bias', 'N/A')}")
    print(f"    lsseg_lr_ratio={params.get('lsseg_lr_ratio', 'N/A')}, batch_size={params.get('batch_size', 'N/A')}")

# 参数统计
print("\n参数统计:")
if e2e_best_params:
    param_keys = list(e2e_best_params[0][2].keys())
    for key in param_keys:
        values = [p[2].get(key) for p in e2e_best_params if key in p[2]]
        if values:
            if isinstance(values[0], (int, float)):
                avg = sum(values) / len(values)
                print(f"  {key}: 平均={avg:.6f}, 范围=[{min(values):.6f}, {max(values):.6f}]")
            else:
                from collections import Counter
                counter = Counter(values)
                print(f"  {key}: 分布={dict(counter)}")

# 汇总建议
print("\n" + "=" * 70)
print("参数固定建议")
print("=" * 70)

print("""
基于 optuna 历史最佳参数，建议固定以下参数：

| 参数 | 建议值 | 原因 |
|------|--------|------|
| batch_size | 4 | 稳定，显存允许 |
| weight_decay | 2e-4 | 平均值，合理正则化 |
| lr_factor | 0.7 | 平均值，学习率衰减稳定 |
| step_size | 15 | 偏大，更平滑的衰减 |
| lora_alpha_ratio | 2 | 固定，alpha = r * 2 |
| freeze_lsseg | False | 端到端训练 |

保持搜索的参数：
| 参数 | 建议范围 | 原因 |
|------|----------|------|
| lr | [1e-5, 1e-3] | 最关键参数 |
| lora_r | [4, 8] | 核心适配参数 |
| prompt_bias | [0.5, 1.5] | 核心优化参数 |
| lsseg_lr_ratio | [0.1, 0.2] | 分层学习率 |
""")

conn.close()