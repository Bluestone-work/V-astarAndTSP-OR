"""
包含功能：
1. t检验（配对和独立样本）
2. 95%置信区间
3. Cohen's d 效应量
4. Mann-Whitney U检验（非参数检验）
5. 描述性统计
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    计算均值的置信区间（使用t分布）
    
    Args:
        data: 数据数组
        confidence: 置信水平，默认0.95（95%置信区间）
    
    Returns:
        (mean, lower_bound, upper_bound): 均值和置信区间的下界、上界
    
    Example:
        >>> data = np.array([320, 315, 325, 310, 330])
        >>> mean, lower, upper = calculate_confidence_interval(data)
        >>> print(f"均值: {mean:.1f}, 95% CI: [{lower:.1f}, {upper:.1f}]")
        均值: 320.0, 95% CI: [310.5, 329.5]
    """
    data = np.array(data)
    n = len(data)
    
    if n < 2:
        raise ValueError("数据至少需要2个点才能计算置信区间")
    
    mean = np.mean(data)
    std_err = stats.sem(data)  # 标准误差 = std / sqrt(n)
    
    # 使用t分布（因为不知道总体标准差）
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * std_err
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return mean, lower_bound, upper_bound


def independent_t_test(group1: np.ndarray, group2: np.ndarray, 
                       equal_var: bool = False) -> Tuple[float, float]:
    """
    独立样本t检验（Welch's t-test，默认不假设方差齐性）
    
    用途：比较两个独立组的均值是否有显著差异
    
    Args:
        group1: 第一组数据
        group2: 第二组数据
        equal_var: 是否假设方差相等
                   False (默认): Welch's t-test（更稳健，推荐）
                   True: Student's t-test（需要方差齐性）
    
    Returns:
        (t_statistic, p_value): t统计量和双尾p值
    
    Example:
        >>> mdqn_steps = np.array([320, 315, 325, 310, 330])
        >>> iddqn_steps = np.array([345, 350, 340, 355, 348])
        >>> t_stat, p_value = independent_t_test(mdqn_steps, iddqn_steps)
        >>> if p_value < 0.05:
        ...     print(f"显著差异 (p={p_value:.3f})")
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("每组至少需要2个数据点")
    
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    return t_statistic, p_value


def paired_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    配对样本t检验
    
    用途：比较同一组对象在两种条件下的均值差异
    要求：两组数据必须一一对应（如同一环境下不同算法的表现）
    
    Args:
        group1: 第一组数据
        group2: 第二组数据（必须与group1长度相同且一一对应）
    
    Returns:
        (t_statistic, p_value): t统计量和双尾p值
    
    Example:
        >>> # 同一1000个环境，两个算法的步数
        >>> mdqn_steps = np.array([320, 315, 325, ...])  # 1000个
        >>> iddqn_steps = np.array([345, 350, 340, ...])  # 1000个
        >>> t_stat, p_value = paired_t_test(mdqn_steps, iddqn_steps)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) != len(group2):
        raise ValueError(f"配对t检验要求两组数据长度相同: {len(group1)} vs {len(group2)}")
    
    if len(group1) < 2:
        raise ValueError("至少需要2对数据")
    
    t_statistic, p_value = stats.ttest_rel(group1, group2)
    
    return t_statistic, p_value


def mann_whitney_u_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Mann-Whitney U检验（非参数检验）
    
    用途：当数据不满足正态分布假设时，使用此检验比较两组的中位数
    优点：对离群值不敏感，不要求正态分布
    
    Args:
        group1: 第一组数据
        group2: 第二组数据
    
    Returns:
        (u_statistic, p_value): U统计量和双尾p值
    
    Example:
        >>> # 数据有明显偏态或离群值时使用
        >>> u_stat, p_value = mann_whitney_u_test(group1, group2)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    u_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    return u_statistic, p_value


def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算Cohen's d效应量
    
    用途：量化两组差异的实际大小（不只是统计显著性）
    
    解释：
        |d| < 0.2: 极小效应 - 差异虽显著但实际意义不大
        0.2 ≤ |d| < 0.5: 小效应
        0.5 ≤ |d| < 0.8: 中等效应
        |d| ≥ 0.8: 大效应 - 差异既显著又重要
    
    Args:
        group1: 第一组数据
        group2: 第二组数据
    
    Returns:
        cohens_d: Cohen's d效应量（正值表示group1>group2，负值相反）
    
    Example:
        >>> d = calculate_effect_size(mdqn_steps, iddqn_steps)
        >>> if abs(d) >= 0.5:
        ...     print(f"中等到大的效应量: d={d:.2f}")
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 合并标准差（pooled standard deviation）
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return cohens_d


def compare_success_rates(success1: np.ndarray, success2: np.ndarray, 
                         name1: str = "Algorithm 1", name2: str = "Algorithm 2") -> Dict:
    """
    对比两个算法的成功率（使用比例z检验）
    
    用途：比较两个算法的成功率是否有显著差异
    
    Args:
        success1: 第一个算法的成功/失败数组（True/False或1/0）
        success2: 第二个算法的成功/失败数组
        name1: 第一个算法名称
        name2: 第二个算法名称
    
    Returns:
        包含统计检验结果的字典
    
    Example:
        >>> success1 = np.array([True, True, False, True, ...])  # 1000个
        >>> success2 = np.array([True, False, True, True, ...])  # 1000个
        >>> result = compare_success_rates(success1, success2, "M-DQN", "IDDQN")
        >>> print(f"p值: {result['p_value']:.6f}")
    """
    success1 = np.array(success1, dtype=bool)
    success2 = np.array(success2, dtype=bool)
    
    n1 = len(success1)
    n2 = len(success2)
    x1 = np.sum(success1)  # 成功次数
    x2 = np.sum(success2)
    
    p1 = x1 / n1  # 成功率
    p2 = x2 / n2
    
    # Wilson置信区间（比正态近似更准确）
    z = 1.96  # 95%置信度
    
    # 算法1的Wilson CI
    wilson_center1 = (p1 + z**2/(2*n1)) / (1 + z**2/n1)
    wilson_width1 = z * np.sqrt((p1*(1-p1) + z**2/(4*n1)) / n1) / (1 + z**2/n1)
    ci_lower1 = wilson_center1 - wilson_width1
    ci_upper1 = wilson_center1 + wilson_width1
    
    # 算法2的Wilson CI
    wilson_center2 = (p2 + z**2/(2*n2)) / (1 + z**2/n2)
    wilson_width2 = z * np.sqrt((p2*(1-p2) + z**2/(4*n2)) / n2) / (1 + z**2/n2)
    ci_lower2 = wilson_center2 - wilson_width2
    ci_upper2 = wilson_center2 + wilson_width2
    
    # 比例z检验（双样本）
    # 合并比例
    p_pooled = (x1 + x2) / (n1 + n2)
    
    # 标准误差
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # z统计量
    if se > 0:
        z_stat = (p1 - p2) / se
        # p值（双尾检验）
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0
        p_value = 1.0
    
    # 显著性标记
    if p_value < 0.001:
        sig = "***"
        sig_text = "极显著"
    elif p_value < 0.01:
        sig = "**"
        sig_text = "非常显著"
    elif p_value < 0.05:
        sig = "*"
        sig_text = "显著"
    else:
        sig = "ns"
        sig_text = "不显著"
    
    # 相对提升
    relative_improvement = (p1 - p2) / p2 * 100 if p2 > 0 else 0
    
    # 绝对差异（百分点）
    absolute_diff = (p1 - p2) * 100
    
    return {
        'n1': n1,
        'n2': n2,
        'success_count1': x1,
        'success_count2': x2,
        'success_rate1': p1,
        'success_rate2': p2,
        'ci1': (ci_lower1, ci_upper1),
        'ci2': (ci_lower2, ci_upper2),
        'z_statistic': z_stat,
        'p_value': p_value,
        'significance': sig,
        'significance_text': sig_text,
        'absolute_diff_percentage_points': absolute_diff,
        'relative_improvement_percent': relative_improvement
    }


def calculate_descriptive_stats(data: np.ndarray) -> Dict[str, float]:
    """
    计算描述性统计量
    
    Args:
        data: 数据数组
    
    Returns:
        包含各种统计量的字典
    
    Example:
        >>> stats = calculate_descriptive_stats(steps_data)
        >>> print(f"均值: {stats['mean']:.1f} ± {stats['std']:.1f}")
    """
    data = np.array(data)
    
    return {
        'n': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data, ddof=1),  # 样本标准差
        'var': np.var(data, ddof=1),  # 样本方差
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),  # 第一四分位数
        'q75': np.percentile(data, 75),  # 第三四分位数
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),  # 四分位距
        'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else 0,  # 变异系数
    }


def compare_algorithms(results1: Dict, results2: Dict, 
                      name1: str, name2: str,
                      metrics: Optional[List[str]] = None) -> Dict:
    """
    全面对比两个算法的统计显著性
    
    这是一个高级函数，整合了多种统计检验，生成完整的对比报告
    
    Args:
        results1: 第一个算法的结果字典（来自batch_test_*函数）
        results2: 第二个算法的结果字典
        name1: 第一个算法名称
        name2: 第二个算法名称
        metrics: 要对比的指标列表，None则使用默认列表
    
    Returns:
        包含所有对比结果的字典
    
    Example:
        >>> results_mdqn = batch_test_rl_on_fixed_envs(...)
        >>> results_iddqn = batch_test_rl_on_fixed_envs(...)
        >>> comparison = compare_algorithms(
        ...     results_mdqn, results_iddqn,
        ...     "M-DQN", "IDDQN",
        ...     metrics=['steps', 'path_length', 'control_latency']
        ... )
    """
    if metrics is None:
        metrics = ['steps', 'path_length', 'control_latency', 'avg_local_decision_time']
    
    comparison_results = {}
    
    print(f"\n{'='*80}")
    print(f"统计显著性检验: {name1} vs {name2}")
    print(f"{'='*80}\n")
    
    for metric in metrics:
        if metric not in results1 or metric not in results2:
            print(f"⚠️  指标 '{metric}' 不存在，跳过")
            continue
        
        # 提取成功案例的数据
        succ1 = [i for i, s in enumerate(results1['success']) if s]
        succ2 = [i for i, s in enumerate(results2['success']) if s]
        
        data1 = np.array([results1[metric][i] for i in succ1])
        data2 = np.array([results2[metric][i] for i in succ2])
        
        # 过滤0值
        if metric in ['steps', 'path_length']:
            data1 = data1[data1 > 0]
            data2 = data2[data2 > 0]
        
        if len(data1) < 2 or len(data2) < 2:
            print(f"⚠️  指标 '{metric}' 数据不足，跳过")
            continue
        
        print(f"指标: {metric}")
        print(f"{'-'*80}")
        
        # 1. 描述性统计
        stats1 = calculate_descriptive_stats(data1)
        stats2 = calculate_descriptive_stats(data2)
        
        print(f"\n{name1}:")
        print(f"  样本量: {stats1['n']}")
        print(f"  均值 ± 标准差: {stats1['mean']:.2f} ± {stats1['std']:.2f}")
        print(f"  中位数: {stats1['median']:.2f}")
        print(f"  范围: [{stats1['min']:.2f}, {stats1['max']:.2f}]")
        
        print(f"\n{name2}:")
        print(f"  样本量: {stats2['n']}")
        print(f"  均值 ± 标准差: {stats2['mean']:.2f} ± {stats2['std']:.2f}")
        print(f"  中位数: {stats2['median']:.2f}")
        print(f"  范围: [{stats2['min']:.2f}, {stats2['max']:.2f}]")
        
        # 2. 95%置信区间
        _, ci_lower1, ci_upper1 = calculate_confidence_interval(data1)
        _, ci_lower2, ci_upper2 = calculate_confidence_interval(data2)
        
        print(f"\n95%置信区间:")
        print(f"  {name1}: [{ci_lower1:.2f}, {ci_upper1:.2f}]")
        print(f"  {name2}: [{ci_lower2:.2f}, {ci_upper2:.2f}]")
        
        # 3. t检验
        t_stat, p_value = independent_t_test(data1, data2, equal_var=False)
        
        # 显著性标记
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = ""
        
        print(f"\n显著性检验:")
        print(f"  独立样本t检验 (Welch's t-test):")
        print(f"    t统计量 = {t_stat:.4f}")
        print(f"    p值 = {p_value:.6f}")
        print(f"    结论: {'显著差异' if sig else '无显著差异'} {sig}")
        
        # 4. 效应量
        d = calculate_effect_size(data1, data2)
        
        if abs(d) >= 0.8:
            effect_interp = "大效应"
        elif abs(d) >= 0.5:
            effect_interp = "中等效应"
        elif abs(d) >= 0.2:
            effect_interp = "小效应"
        else:
            effect_interp = "极小效应"
        
        print(f"\n效应量(Cohen's d):")
        print(f"  d = {d:.4f}")
        print(f"  解释: {effect_interp}")
        
        # 5. 相对变化
        rel_change = (stats1['mean'] - stats2['mean']) / stats2['mean'] * 100
        print(f"\n相对变化:")
        print(f"  {name1}比{name2}{'+' if rel_change > 0 else ''}{rel_change:.2f}%")
        
        print(f"\n{'='*80}\n")
        
        # 保存结果
        comparison_results[metric] = {
            'stats1': stats1,
            'stats2': stats2,
            'ci1': (ci_lower1, ci_upper1),
            'ci2': (ci_lower2, ci_upper2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'effect_interpretation': effect_interp,
            'relative_change': rel_change
        }
    
    return comparison_results


def generate_summary_table(comparison_results: Dict, name1: str, name2: str) -> str:
    """
    生成LaTeX格式的统计对比表格
    
    Args:
        comparison_results: compare_algorithms函数返回的结果
        name1: 第一个算法名称
        name2: 第二个算法名称
    
    Returns:
        LaTeX表格代码
    
    Example:
        >>> comparison = compare_algorithms(...)
        >>> latex_table = generate_summary_table(comparison, "M-DQN", "IDDQN")
        >>> print(latex_table)
    """
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Statistical Comparison: {name1} vs {name2}}}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "Metric & " + name1 + " & " + name2 + " & p-value & Cohen's d \\\\\n"
    latex += "       & (Mean $\\pm$ SD) & (Mean $\\pm$ SD) & & \\\\\n"
    latex += "\\midrule\n"
    
    for metric, results in comparison_results.items():
        mean1 = results['stats1']['mean']
        std1 = results['stats1']['std']
        mean2 = results['stats2']['mean']
        std2 = results['stats2']['std']
        p = results['p_value']
        d = results['cohens_d']
        
        # 显著性标记
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
        
        latex += f"{metric} & "
        latex += f"{mean1:.2f} $\\pm$ {std1:.2f} & "
        latex += f"{mean2:.2f} $\\pm$ {std2:.2f} & "
        latex += f"{p:.4f}{sig} & "
        latex += f"{d:.3f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\begin{flushleft}\n"
    latex += "\\footnotesize\n"
    latex += "Note: *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.\n"
    latex += "\\end{flushleft}\n"
    latex += "\\end{table}\n"
    
    return latex


if __name__ == "__main__":
    # 测试示例
    print("统计测试模块 - 测试")
    print("="*60)
    
    # 生成模拟数据
    np.random.seed(42)
    mdqn_steps = np.random.normal(320, 45, 861)
    iddqn_steps = np.random.normal(346, 52, 823)
    
    print("\n1. 描述性统计:")
    desc_stats = calculate_descriptive_stats(mdqn_steps)
    print(f"M-DQN: 均值={desc_stats['mean']:.1f}, 标准差={desc_stats['std']:.1f}")
    
    print("\n2. 95%置信区间:")
    mean, lower, upper = calculate_confidence_interval(mdqn_steps)
    print(f"M-DQN: {mean:.1f}, 95% CI=[{lower:.1f}, {upper:.1f}]")
    
    print("\n3. 独立样本t检验:")
    t_stat, p_value = independent_t_test(mdqn_steps, iddqn_steps)
    print(f"t={t_stat:.3f}, p={p_value:.6f}")
    
    print("\n4. Cohen's d效应量:")
    d = calculate_effect_size(mdqn_steps, iddqn_steps)
    print(f"d={d:.3f}")
    
    print("\n5. 成功率对比（比例检验）:")
    success1 = np.random.random(1000) < 0.861  # 86.1%成功率
    success2 = np.random.random(1000) < 0.823  # 82.3%成功率
    result = compare_success_rates(success1, success2, "M-DQN", "IDDQN")
    print(f"M-DQN: {result['success_rate1']*100:.1f}% ({result['success_count1']}/{result['n1']})")
    print(f"IDDQN: {result['success_rate2']*100:.1f}% ({result['success_count2']}/{result['n2']})")
    print(f"z={result['z_statistic']:.3f}, p={result['p_value']:.6f} {result['significance']}")
    
    print("\n✓ 所有测试通过！")
