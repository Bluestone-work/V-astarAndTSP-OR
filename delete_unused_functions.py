#!/usr/bin/env python3
"""删除evaluate.py中未使用的函数"""

import re

# 读取文件
with open('/home/wangjiang/copy/V-A*_and_TSP-OR/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义要删除的函数列表
functions_to_delete = [
    'batch_test_traditional_on_fixed_envs',
    'compare_all_on_random_envs',
    '_test_algorithm_random_envs',
    'batch_test_traditional_on_random_envs',
    '_perform_statistical_tests',
    '_summarize_and_save_random_env_results',
    '_plot_random_env_comparison',
    'compare_all_on_fixed_envs',
    '_summarize_and_save_fixed_env_results',
    '_plot_fixed_env_comparison',
    'run_swrdqn_without_viz_old',
    'plot_trajectory_comparison',
    'generate_trajectory_gif',
    'batch_test_rl_on_fixed_envs_auto',
    'run_traditional_with_gif',
    'run_sb3_ppo_without_viz',
    'run_navigation_without_viz',
]

print("开始删除未使用的函数...")
print(f"文件原始大小: {len(content)} 字符")

for func_name in functions_to_delete:
    # 匹配函数定义到下一个def或文件末尾
    # 使用非贪婪匹配,找到函数开始到下一个同级别def的位置
    pattern = rf'^def {re.escape(func_name)}\([^)]*\).*?(?=^def |\Z)'
    
    matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
    
    if matches:
        print(f"  删除函数: {func_name} ({len(matches[0].group())} 字符)")
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    else:
        print(f"  未找到函数: {func_name}")

print(f"\n文件删除后大小: {len(content)} 字符")
print(f"减少了: {len(content) - len(open('/home/wangjiang/copy/V-A*_and_TSP-OR/evaluate.py', 'r').read())} 字符")

# 写入新文件
output_path = '/home/wangjiang/copy/V-A*_and_TSP-OR/evaluate_cleaned.py'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n已保存到: {output_path}")
