import sqlite3
import json

# 数据库路径
db_path = '/home/wh/projects/lssegv2.1-master/optuna.db'

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 定义要分析的 studies
target_studies = [
    # LSSeg baseline (02-14)
    (6, 'test_RITE 02-14 08_30 fold_0'),
    (7, 'test_RITE 02-14 08_30 fold_1'),
    (8, 'test_RITE 02-14 08_30 fold_2'),
    (9, 'test_RITE 02-14 08_30 fold_3'),
    (10, 'test_RITE 02-14 08_30 fold_4'),
    # LSSeg+SamLoRA 端到端 (03-09)
    (182, 'test_RITE_lsseg+samlora+Prompt Bias_端到端_new-parameters 03-09 16_23 fold_0'),
    (183, 'test_RITE_lsseg+samlora+Prompt Bias_端到端_new-parameters 03-09 16_23 fold_1'),
    (184, 'test_RITE_lsseg+samlora+Prompt Bias_端到端_new-parameters 03-09 16_23 fold_2'),
    (185, 'test_RITE_lsseg+samlora+Prompt Bias_端到端_new-parameters 03-09 16_23 fold_3'),
    (186, 'test_RITE_lsseg+samlora+Prompt Bias_端到端_new-parameters 03-09 16_23 fold_4'),
]

for study_id, study_name in target_studies:
    print(f"\n{'='*70}")
    print(f"Study: {study_name}")
    print('='*70)
    
    # 查询 trials 和 values
    cursor.execute('''
        SELECT t.trial_id, t.number, t.state, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ?
        ORDER BY t.number
    ''', (study_id,))
    trials = cursor.fetchall()
    
    if len(trials) == 0:
        print("  No trials found.")
        continue
    
    # 统计完成的试验
    completed = [(tid, num, val) for tid, num, state, val in trials if state == 'COMPLETE' and val is not None]
    print(f"  总试验数: {len(trials)}")
    print(f"  完成试验数: {len(completed)}")
    
    if completed:
        # 找最佳 trial
        best = max(completed, key=lambda x: x[2])
        print(f"  最佳 Dice 值: {best[2]:.4f}")
        
        # 查询最佳 trial 的参数
        cursor.execute('''
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        ''', (best[0],))
        params = cursor.fetchall()
        
        print(f"\n  最佳参数配置 (Trial {best[1]}):")
        for param_name, param_value in params:
            try:
                value = json.loads(param_value)
            except:
                value = param_value
            print(f"    {param_name}: {value}")
        
        # 显示所有完成 trials 的概览
        print(f"\n  所有完成试验的 Top 3:")
        sorted_completed = sorted(completed, key=lambda x: x[2], reverse=True)[:3]
        for tid, num, val in sorted_completed:
            cursor.execute('''
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            ''', (tid,))
            params = cursor.fetchall()
            params_dict = {}
            for pn, pv in params:
                try:
                    params_dict[pn] = json.loads(pv)
                except:
                    params_dict[pn] = pv
            print(f"    Trial {num}: Dice={val:.4f}")
            for k, v in params_dict.items():
                print(f"      {k}: {v}")

# 汇总对比分析
print(f"\n\n{'='*70}")
print("汇总对比分析")
print('='*70)

# LSSeg baseline
print("\n【LSSeg Baseline (02-14 08_30) 最佳参数汇总】")
lsseg_folds = [(6, 'fold_0'), (7, 'fold_1'), (8, 'fold_2'), (9, 'fold_3'), (10, 'fold_4')]
lsseg_best_params = []
for study_id, fold_name in lsseg_folds:
    cursor.execute('''
        SELECT t.trial_id, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ORDER BY tv.value DESC
        LIMIT 1
    ''', (study_id,))
    result = cursor.fetchone()
    if result:
        trial_id, value = result
        cursor.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id = ?', (trial_id,))
        params = {pn: json.loads(pv) if pv.startswith('[') or pv.startswith('{') else pv 
                  for pn, pv in cursor.fetchall()}
        lsseg_best_params.append((fold_name, value, params))
        print(f"  {fold_name}: Dice={value:.4f}, lr={params.get('lr', 'N/A')}, batch_size={params.get('batch_size', 'N/A')}")

# LSSeg+SamLoRA 端到端
print("\n【LSSeg+SamLoRA 端到端 (03-09 16_23) 最佳参数汇总】")
e2e_folds = [(182, 'fold_0'), (183, 'fold_1'), (184, 'fold_2'), (185, 'fold_3'), (186, 'fold_4')]
e2e_best_params = []
for study_id, fold_name in e2e_folds:
    cursor.execute('''
        SELECT t.trial_id, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ORDER BY tv.value DESC
        LIMIT 1
    ''', (study_id,))
    result = cursor.fetchone()
    if result:
        trial_id, value = result
        cursor.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id = ?', (trial_id,))
        params = {pn: json.loads(pv) if pv.startswith('[') or pv.startswith('{') else pv 
                  for pn, pv in cursor.fetchall()}
        e2e_best_params.append((fold_name, value, params))
        print(f"  {fold_name}: Dice={value:.4f}")
        print(f"    lr={params.get('lr', 'N/A'):.6f}, lora_r={params.get('lora_r', 'N/A')}, prompt_bias={params.get('prompt_bias', 'N/A')}")
        print(f"    lsseg_lr_ratio={params.get('lsseg_lr_ratio', 'N/A')}")

conn.close()
print("\n分析完成!")