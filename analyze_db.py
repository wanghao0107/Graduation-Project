import sqlite3
import json
import os

# 数据库路径
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optuna.db')

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询所有 studies
cursor.execute('SELECT study_id, study_name FROM studies')
studies = cursor.fetchall()

print("=" * 80)
print("所有 Studies 列表:")
print("=" * 80)
for study_id, study_name in studies:
    print(f"  [{study_id}] {study_name}")

# 分析最近的两个实验
target_studies = [
    'test_RITE_LSSeg 02-14 08_30',
    'test_RITE_lsseg+samlora+Prompt Bias_端到端_new-parameters 03-09 16_23'
]

print("\n" + "=" * 80)
print("详细分析目标实验:")
print("=" * 80)

for study_name_pattern in target_studies:
    print(f"\n{'='*70}")
    print(f"实验: {study_name_pattern}")
    print('='*70)
    
    # 查找匹配的 studies
    cursor.execute('SELECT study_id, study_name FROM studies WHERE study_name LIKE ?', 
                   (f'%{study_name_pattern}%',))
    matching_studies = cursor.fetchall()
    
    all_best_params = []
    
    for study_id, study_name in matching_studies:
        # 查询最佳 trial
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
            all_best_params.append((study_name, value, params))
    
    if all_best_params:
        # 汇总最佳参数
        print(f"\n各折最佳结果:")
        for name, value, params in all_best_params:
            print(f"  {name.split()[-1]}: Dice={value:.4f}")
        
        # 统计参数分布
        print(f"\n参数统计:")
        param_keys = all_best_params[0][2].keys()
        for key in param_keys:
            values = [p[2].get(key) for p in all_best_params if key in p[2]]
            if values:
                if isinstance(values[0], (int, float)):
                    avg = sum(values) / len(values)
                    print(f"  {key}: 平均={avg:.6f}, 范围=[{min(values):.6f}, {max(values):.6f}]")
                else:
                    from collections import Counter
                    counter = Counter(values)
                    print(f"  {key}: 分布={dict(counter)}")

conn.close()
print("\n分析完成!")