"""
主运行脚本 - 依次运行所有5个问题
"""
import sys
import os
import time

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from problem1 import solve_problem1
from problem2 import solve_problem2
from problem3 import solve_problem3
from problem4 import solve_problem4
from problem5 import solve_problem5


def main():
    print("=" * 70)
    print("   C题 烟幕干扰弹的投放策略 - Python求解")
    print("   基于2025年A题解法，使用moni2026 C题参数")
    print("=" * 70)
    print()
    print("关键参数差异 (C题 vs A题):")
    print("  烟幕云团下沉速度: 2.5 m/s (原 3.0 m/s)")
    print("  无人机速度范围: 80~120 m/s (原 70~140 m/s)")
    print("  问题1投放时间: 1.2 s (原 1.5 s)")
    print("  问题1起爆延时: 3.2 s (原 3.6 s)")
    print()

    results = {}
    total_start = time.time()

    # 问题1: 固定参数验证
    print("\n" + "#" * 70)
    print("# 问题1: 固定参数，计算有效遮蔽时长")
    print("#" * 70)
    t1_start = time.time()
    t1 = solve_problem1()
    t1_elapsed = time.time() - t1_start
    results['problem1'] = {'time': t1, 'elapsed': t1_elapsed}
    print(f"\n[问题1 完成, 耗时: {t1_elapsed:.1f}s]")

    # 问题2: 单机单弹优化
    print("\n" + "#" * 70)
    print("# 问题2: 单机单弹最优投放策略")
    print("#" * 70)
    t2_start = time.time()
    x2, t2 = solve_problem2()
    t2_elapsed = time.time() - t2_start
    results['problem2'] = {'x': x2, 'time': t2, 'elapsed': t2_elapsed}
    print(f"\n[问题2 完成, 耗时: {t2_elapsed:.1f}s]")

    # 问题3: 单机三弹优化
    print("\n" + "#" * 70)
    print("# 问题3: 单机三弹最优投放策略 (result1.xlsx)")
    print("#" * 70)
    t3_start = time.time()
    x3, t3 = solve_problem3()
    t3_elapsed = time.time() - t3_start
    results['problem3'] = {'x': x3, 'time': t3, 'elapsed': t3_elapsed}
    print(f"\n[问题3 完成, 耗时: {t3_elapsed:.1f}s]")

    # 问题4: 三机各一弹协同
    print("\n" + "#" * 70)
    print("# 问题4: 三机协同最优投放策略 (result2.xlsx)")
    print("#" * 70)
    t4_start = time.time()
    x4, t4 = solve_problem4()
    t4_elapsed = time.time() - t4_start
    results['problem4'] = {'x': x4, 'time': t4, 'elapsed': t4_elapsed}
    print(f"\n[问题4 完成, 耗时: {t4_elapsed:.1f}s]")

    # 问题5: 五机多弹多导弹协同
    print("\n" + "#" * 70)
    print("# 问题5: 五机多弹协同最优投放策略 (result3.xlsx)")
    print("#" * 70)
    t5_start = time.time()
    res5, t5 = solve_problem5()
    t5_elapsed = time.time() - t5_start
    results['problem5'] = {'result': res5, 'time': t5, 'elapsed': t5_elapsed}
    print(f"\n[问题5 完成, 耗时: {t5_elapsed:.1f}s]")

    # 汇总
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("                      最终结果汇总")
    print("=" * 70)
    print(f"  问题1 (固定参数验证):    {results['problem1']['time']:.4f} s")
    print(f"  问题2 (单机单弹最优):    {results['problem2']['time']:.4f} s")
    print(f"  问题3 (单机三弹最优):    {results['problem3']['time']:.4f} s  → result1.xlsx")
    print(f"  问题4 (三机协同最优):    {results['problem4']['time']:.4f} s  → result2.xlsx")
    print(f"  问题5 (五机多弹协同):    {results['problem5']['time']:.4f} s  → result3.xlsx")
    print(f"\n  总运行时间: {total_elapsed:.1f} s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    # 对比参考值
    print("\n参考值 (2025年A题原参数):")
    print("  问题1: 1.3915 s (原参数: 延时1.5/3.6s)")
    print("  问题2: 4.5960 s")
    print("  问题3: 7.6500 s")
    print("  问题4: 11.7540 s")
    print("  问题5: 38.0600 s")
    print("  (C题参数下沉速度更慢2.5 vs 3.0m/s，遮蔽时间可能略长)")


if __name__ == "__main__":
    main()
