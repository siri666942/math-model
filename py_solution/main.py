"""
主运行脚本 - 依次运行所有5个问题，或通过命令行参数指定只运行其中若干个

用法:
    python main.py            # 依次运行全部5个问题（不传参数，等价于原有行为）
    python main.py 1          # 只运行问题1
    python main.py 2 4        # 只运行问题2和问题4
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


def run_problem1(results):
    print("\n" + "#" * 70)
    print("# 问题1: 固定参数，计算有效遮蔽时长")
    print("#" * 70)
    t_start = time.time()
    t1 = solve_problem1()
    elapsed = time.time() - t_start
    results['problem1'] = {'time': t1, 'elapsed': elapsed}
    print(f"\n[问题1 完成, 耗时: {elapsed:.1f}s]")


def run_problem2(results):
    print("\n" + "#" * 70)
    print("# 问题2: 单机单弹最优投放策略")
    print("#" * 70)
    t_start = time.time()
    x2, t2 = solve_problem2()
    elapsed = time.time() - t_start
    results['problem2'] = {'x': x2, 'time': t2, 'elapsed': elapsed}
    print(f"\n[问题2 完成, 耗时: {elapsed:.1f}s]")


def run_problem3(results):
    print("\n" + "#" * 70)
    print("# 问题3: 单机三弹最优投放策略 (result1.xlsx)")
    print("#" * 70)
    t_start = time.time()
    x3, t3 = solve_problem3()
    elapsed = time.time() - t_start
    results['problem3'] = {'x': x3, 'time': t3, 'elapsed': elapsed}
    print(f"\n[问题3 完成, 耗时: {elapsed:.1f}s]")


def run_problem4(results):
    print("\n" + "#" * 70)
    print("# 问题4: 三机协同最优投放策略 (result2.xlsx)")
    print("#" * 70)
    t_start = time.time()
    x4, t4 = solve_problem4()
    elapsed = time.time() - t_start
    results['problem4'] = {'x': x4, 'time': t4, 'elapsed': elapsed}
    print(f"\n[问题4 完成, 耗时: {elapsed:.1f}s]")


def run_problem5(results):
    print("\n" + "#" * 70)
    print("# 问题5: 五机多弹协同最优投放策略 (result3.xlsx)")
    print("#" * 70)
    t_start = time.time()
    res5, t5 = solve_problem5()
    elapsed = time.time() - t_start
    results['problem5'] = {'result': res5, 'time': t5, 'elapsed': elapsed}
    print(f"\n[问题5 完成, 耗时: {elapsed:.1f}s]")


# 问题编号 -> 对应的执行函数，供命令行参数选择时查找
RUNNERS = {1: run_problem1, 2: run_problem2, 3: run_problem3, 4: run_problem4, 5: run_problem5}


def parse_selection(argv):
    """不传参数则跑全部；传数字则只跑对应问题，例如 `python main.py 1 3`"""
    if not argv:
        return [1, 2, 3, 4, 5]
    try:
        selected = sorted({int(a) for a in argv})
    except ValueError:
        print("用法: python main.py [问题编号 ...]   例如: python main.py 1 3")
        print("不带参数则依次运行全部5个问题。")
        sys.exit(1)
    invalid = [n for n in selected if n not in RUNNERS]
    if invalid:
        print(f"问题编号只能是 1~5，收到无效值: {invalid}")
        sys.exit(1)
    return selected


def main():
    selected = parse_selection(sys.argv[1:])

    print("=" * 70)
    print("   C题 烟幕干扰弹的投放策略 - Python求解")
    print("   基于2025年A题解法，使用moni2026 C题参数")
    print(f"   本次运行问题: {selected}")
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
    for n in selected:
        RUNNERS[n](results)
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("                      本次运行结果汇总")
    print("=" * 70)
    if 'problem1' in results:
        print(f"  问题1 (固定参数验证):    {results['problem1']['time']:.4f} s")
    if 'problem2' in results:
        print(f"  问题2 (单机单弹最优):    {results['problem2']['time']:.4f} s")
    if 'problem3' in results:
        print(f"  问题3 (单机三弹最优):    {results['problem3']['time']:.4f} s  → result1.xlsx")
    if 'problem4' in results:
        print(f"  问题4 (三机协同最优):    {results['problem4']['time']:.4f} s  → result2.xlsx")
    if 'problem5' in results:
        print(f"  问题5 (五机多弹协同):    {results['problem5']['time']:.4f} s  → result3.xlsx")
    print(f"\n  总运行时间: {total_elapsed:.1f} s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    if selected == [1, 2, 3, 4, 5]:
        print("\n参考值 (2025年A题原参数):")
        print("  问题1: 1.3915 s (原参数: 延时1.5/3.6s)")
        print("  问题2: 4.5960 s")
        print("  问题3: 7.6500 s")
        print("  问题4: 11.7540 s")
        print("  问题5: 38.0600 s")
        print("  (C题参数下沉速度更慢2.5 vs 3.0m/s，遮蔽时间可能略长)")


if __name__ == "__main__":
    main()
