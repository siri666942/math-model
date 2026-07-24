"""
主驱动脚本 - 运行所有5个问题的求解并对比论文结果
1. 使用 config_a (A题原参数) → 与论文cumcm25028结果对比
2. 使用 config_c (C题修改参数) → 生成新结果

用法:
    python run_all.py
"""
import numpy as np
import time
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation as sim
from solve_p1 import solve_p1
from solve_p2 import solve_p2
from solve_p3 import solve_p3
from solve_p4 import solve_p4
from solve_p5 import solve_p5


def print_separator(title):
    """打印分隔标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_header():
    """打印程序头部信息"""
    print("=" * 70)
    print("  烟幕干扰弹投放策略 - GA求解 (基于cumcm25028论文方法)")
    print("  目标离散化: 50圆周点 × 11层 = 550关键点")
    print("  遮蔽判定: 点到直线距离法")
    print("  优化方法: 自实现遗传算法")
    print("=" * 70)


def run_all_with_config(config_module, config_name, results_file):
    """
    使用指定配置运行所有5个问题

    参数:
        config_module: 配置模块
        config_name: 配置名称 (用于打印)
        results_file: 结果输出文件路径
    """
    print_separator(f"使用配置: {config_name}")
    print(f"  烟幕下沉速度: {config_module.SMOKE_SINK_SPEED} m/s")
    print(f"  无人机速度范围: [{config_module.DRONE_SPEED_MIN}, {config_module.DRONE_SPEED_MAX}] m/s")
    print(f"  P1投放时间: {config_module.P1_RELEASE_TIME}s, 起爆延时: {config_module.P1_DETONATION_DELAY}s")
    print(f"  目标关键点数: {config_module.N_CIRCLE_POINTS}×{config_module.N_SIDE_LAYERS} = "
          f"{config_module.N_CIRCLE_POINTS * config_module.N_SIDE_LAYERS}")

    results = {}
    total_start = time.time()

    # ---- 问题1 ----
    print_separator("问题1: 固定参数仿真 (单机单弹)")
    try:
        p1_result = solve_p1(config_module, verbose=True)
        results['P1'] = {'effective_time': p1_result}
    except Exception as e:
        print(f"  [错误] 问题1求解失败: {e}")
        import traceback
        traceback.print_exc()
        results['P1'] = {'effective_time': None, 'error': str(e)}

    # ---- 问题2 ----
    print_separator("问题2: 单机单弹最优策略 (GA: 4变量)")
    try:
        theta, speed, t_rel, t_lag, p2_time = solve_p2(config_module, verbose=True)
        results['P2'] = {
            'theta': theta, 'speed': speed,
            't_rel': t_rel, 't_lag': t_lag,
            'effective_time': p2_time,
        }
    except Exception as e:
        print(f"  [错误] 问题2求解失败: {e}")
        import traceback
        traceback.print_exc()
        results['P2'] = {'effective_time': None, 'error': str(e)}

    # ---- 问题3 ----
    print_separator("问题3: 单机三弹最优策略 (GA: 8变量 + 间隔约束)")
    try:
        theta, speed, t_rels, t_lags, p3_time = solve_p3(config_module, verbose=True)
        results['P3'] = {
            'theta': theta, 'speed': speed,
            't_rels': t_rels, 't_lags': t_lags,
            'effective_time': p3_time,
        }
    except Exception as e:
        print(f"  [错误] 问题3求解失败: {e}")
        import traceback
        traceback.print_exc()
        results['P3'] = {'effective_time': None, 'error': str(e)}

    # ---- 问题4 ----
    print_separator("问题4: 三机协同单弹 (GA: 12变量)")
    try:
        drone_results, p4_time = solve_p4(config_module, verbose=True)
        results['P4'] = {
            'drone_results': drone_results,
            'effective_time': p4_time,
        }
    except Exception as e:
        print(f"  [错误] 问题4求解失败: {e}")
        import traceback
        traceback.print_exc()
        results['P4'] = {'effective_time': None, 'error': str(e)}

    # ---- 问题5 ----
    print_separator("问题5: 五机三弹协同 (GA: 40变量 + 间隔约束)")
    try:
        drone_results, p5_time = solve_p5(config_module, verbose=True)
        results['P5'] = {
            'drone_results': drone_results,
            'effective_time': p5_time,
        }
    except Exception as e:
        print(f"  [错误] 问题5求解失败: {e}")
        import traceback
        traceback.print_exc()
        results['P5'] = {'effective_time': None, 'error': str(e)}

    total_elapsed = time.time() - total_start

    # ---- 汇总 ----
    print_separator(f"{config_name} 结果汇总")
    for p in ['P1', 'P2', 'P3', 'P4', 'P5']:
        if p in results:
            r = results[p]
            if r.get('effective_time') is not None:
                print(f"  {p}: 有效遮蔽时长 = {r['effective_time']:.4f} s")
            else:
                print(f"  {p}: 求解失败 ({r.get('error', 'Unknown')})")
    print(f"\n  总运行时间: {total_elapsed:.1f}s")

    # ---- 写入结果文件 ----
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"烟幕干扰弹投放策略 - GA求解结果\n")
        f.write(f"配置: {config_name}\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"烟幕下沉速度: {config_module.SMOKE_SINK_SPEED} m/s\n")
        f.write(f"无人机速度范围: [{config_module.DRONE_SPEED_MIN}, {config_module.DRONE_SPEED_MAX}] m/s\n")
        f.write(f"P1参数: 投放={config_module.P1_RELEASE_TIME}s, 起爆延时={config_module.P1_DETONATION_DELAY}s\n")
        f.write(f"目标关键点: {config_module.N_CIRCLE_POINTS}×{config_module.N_SIDE_LAYERS} = "
                f"{config_module.N_CIRCLE_POINTS*config_module.N_SIDE_LAYERS}\n")
        f.write(f"时间步长: DT={config_module.DT}s\n")
        f.write("=" * 60 + "\n\n")

        for p in ['P1', 'P2', 'P3', 'P4', 'P5']:
            if p in results:
                r = results[p]
                f.write(f"{p}:\n")
                if r.get('effective_time') is not None:
                    f.write(f"  有效遮蔽时长: {r['effective_time']:.6f} s\n")
                    if p == 'P2':
                        f.write(f"  theta: {r['theta']:.6f} rad ({np.degrees(r['theta']):.4f} deg)\n")
                        f.write(f"  speed: {r['speed']:.4f} m/s\n")
                        f.write(f"  t_rel: {r['t_rel']:.6f} s\n")
                        f.write(f"  t_lag: {r['t_lag']:.6f} s\n")
                    elif p == 'P3':
                        f.write(f"  theta: {r['theta']:.6f} rad ({np.degrees(r['theta']):.4f} deg)\n")
                        f.write(f"  speed: {r['speed']:.4f} m/s\n")
                        for bi in range(3):
                            f.write(f"  弹{bi+1}: t_rel={r['t_rels'][bi]:.6f}s, "
                                    f"t_lag={r['t_lags'][bi]:.6f}s, "
                                    f"起爆={r['t_rels'][bi]+r['t_lags'][bi]:.6f}s\n")
                    elif p == 'P4':
                        for dr in r['drone_results']:
                            f.write(f"  {dr['drone']}->{dr['missile']}: theta={dr['theta']:.6f}rad, "
                                    f"v={dr['speed']:.4f}m/s, "
                                    f"t_rel={dr['t_rel']:.6f}s, t_lag={dr['t_lag']:.6f}s\n")
                    elif p == 'P5':
                        for dr in r['drone_results']:
                            f.write(f"  {dr['drone']}: theta={dr['theta']:.6f}rad, "
                                    f"v={dr['speed']:.4f}m/s, missiles={dr['missiles']}\n")
                            for bi in range(3):
                                f.write(f"    弹{bi+1}->M{dr['missiles'][bi]+1}: "
                                        f"t_rel={dr['t_rels'][bi]:.6f}s, "
                                        f"t_lag={dr['t_lags'][bi]:.6f}s, "
                                        f"起爆={dr['t_rels'][bi]+dr['t_lags'][bi]:.6f}s\n")
                else:
                    f.write(f"  错误: {r.get('error', 'Unknown')}\n")
                f.write("\n")

        f.write(f"\n总运行时间: {total_elapsed:.1f}s\n")

    print(f"\n  结果已保存至: {results_file}")

    return results, total_elapsed


def main():
    print_header()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ========================
    # A题验证
    # ========================
    import config_a
    results_a_file = os.path.join(base_dir, 'results_a.txt')
    results_a, time_a = run_all_with_config(config_a, 'A题 (原参数)', results_a_file)

    # 与论文对比
    print_separator("A题 与论文对比")
    paper_results = {
        'P1': 1.3916,
        'P2': 4.588,
        'P3': 6.400,
        'P4': 12.45,
        'P5': 20.24,
    }
    print(f"  {'Problem':<6s} {'Paper':>10s} {'Ours':>10s} {'Diff':>10s} {'Rel.Error':>10s}")
    print(f"  {'-'*50}")
    for p in ['P1', 'P2', 'P3', 'P4', 'P5']:
        paper_val = paper_results[p]
        our_val = results_a.get(p, {}).get('effective_time')
        if our_val is not None:
            diff = our_val - paper_val
            rel_err = abs(diff) / paper_val * 100 if paper_val > 0 else 0
            print(f"  {p:<6s} {paper_val:10.4f} {our_val:10.4f} {diff:+10.4f} {rel_err:9.2f}%")
        else:
            print(f"  {p:<6s} {paper_val:10.4f} {'FAILED':>10s}")

    # ========================
    # C题求解
    # ========================
    time_c = None
    print_separator("C题 (修改参数) 开始求解")
    try:
        import config_c
        results_c_file = os.path.join(base_dir, 'results_c.txt')
        results_c, time_c = run_all_with_config(config_c, 'C题 (修改参数)', results_c_file)
    except Exception as e:
        print(f"  [错误] C题求解失败: {e}")
        import traceback
        traceback.print_exc()

    print_separator("全部求解完成")
    print(f"  A题总耗时: {time_a:.1f}s")
    if time_c is not None:
        print(f"  C题总耗时: {time_c:.1f}s")


if __name__ == '__main__':
    main()
