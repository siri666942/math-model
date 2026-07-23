"""
论文配图生成 - matplotlib

只做静态图（论文要嵌进Word），风格上遵循几条硬规则：
- 单一坐标轴，不用双y轴
- 分类颜色固定顺序，不循环取色，不用彩虹色
- 线条细、网格淡化、直接标注关键点而不是堆图例
- 颜色本身可能有色弱读者，红/绿这类容易混淆的对比尽量避开

用法: 每个 plot_xxx() 函数返回 (fig, ax)，调用方自己决定 savefig 到哪。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import config as cfg
from simulation import missile_position

# 全局配色：固定语义，不循环
COLOR_MISSILE = "#C0392B"     # 导弹 - 暗红
COLOR_DRONE = "#2E6F9E"       # 无人机 - 蓝
COLOR_TARGET = "#4C4C4C"      # 真目标 - 深灰
COLOR_FAKE = "#8A8A8A"        # 假目标 - 浅灰
COLOR_SMOKE = "#7F8C8D"       # 烟幕云团 - 中性灰(半透明画球体)
COLOR_ACCENT = "#D68910"      # 强调/阈值线 - 橙

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Heiti SC", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_scenario_setup():
    """3D 场景图: 3枚导弹、5架无人机、真/假目标初始位置"""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    M = cfg.MISSILES_INIT
    D = cfg.DRONES_INIT
    missile_names = ["M1", "M2", "M3"]
    drone_names = ["FY1", "FY2", "FY3", "FY4", "FY5"]

    ax.scatter(M[:, 0], M[:, 1], M[:, 2], c=COLOR_MISSILE, marker="^", s=90,
               label="来袭导弹", depthshade=False, zorder=5)
    for i, name in enumerate(missile_names):
        ax.text(M[i, 0] + 400, M[i, 1], M[i, 2], name, color=COLOR_MISSILE, fontsize=9)

    ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=COLOR_DRONE, marker="o", s=70,
               label="无人机", depthshade=False, zorder=5)
    for i, name in enumerate(drone_names):
        ax.text(D[i, 0] + 400, D[i, 1], D[i, 2], name, color=COLOR_DRONE, fontsize=9)

    ax.scatter(*cfg.FAKE_TARGET, c=COLOR_FAKE, marker="x", s=80, label="假目标(原点)", zorder=5)
    ax.scatter(cfg.TARGET_CENTER[0], cfg.TARGET_CENTER[1], cfg.TARGET_CENTER[2],
               c=COLOR_TARGET, marker="s", s=60, label="真目标", zorder=5)

    # 导弹指向假目标的航线(细虚线，弱化，不抢主体)
    for i in range(3):
        ax.plot([M[i, 0], 0], [M[i, 1], 0], [M[i, 2], 0],
                color=COLOR_MISSILE, linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("场景初始布局：导弹 / 无人机 / 真假目标", fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    return fig, ax


def plot_problem1_timeline(dt=0.001, t_total=15.0):
    """
    问题1固定场景的距离-时间关系图：
    导弹-目标距离 vs 导弹-云团距离，随时间变化，标出起爆时刻和(如果有)有效遮蔽区间。
    直接用 config.py 当前的问题1参数计算，不依赖任何后台还在跑的优化结果。
    """
    drone_init = cfg.DRONES_INIT[0]
    theta = cfg.P1_DRONE_THETA
    speed = cfg.P1_DRONE_SPEED
    release_time = cfg.P1_RELEASE_TIME
    delay = cfg.P1_DETONATION_DELAY
    detonation_time = release_time + delay
    smoke_expire = detonation_time + cfg.EFFECTIVE_DURATION

    ts = np.arange(0, t_total, dt)
    M_pos = np.array([missile_position(t, 0) for t in ts])
    dist_mt = np.linalg.norm(M_pos - cfg.TARGET_CENTER, axis=1)

    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    active = ts >= detonation_time
    smoke_pos = np.full((len(ts), 3), np.nan)
    if np.any(active):
        FY_at_release = drone_init + speed * direction * release_time
        bomb_xy = FY_at_release[:2] + speed * direction[:2] * delay
        bomb_h = FY_at_release[2] - 0.5 * cfg.G * delay ** 2
        t_active = ts[active]
        smoke_pos[active, 0] = bomb_xy[0]
        smoke_pos[active, 1] = bomb_xy[1]
        smoke_pos[active, 2] = bomb_h - cfg.SMOKE_SINK_SPEED * (t_active - detonation_time)
        expired = ts > smoke_expire
        smoke_pos[expired] = np.nan

    dist_ms = np.linalg.norm(M_pos - smoke_pos, axis=1)
    valid = ~np.isnan(dist_ms)
    i_min = np.nanargmin(dist_ms)
    t_min, d_min = ts[i_min], dist_ms[i_min]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, dist_mt, color=COLOR_MISSILE, linewidth=1.6, label="导弹—真目标 距离")
    ax.plot(ts, dist_ms, color=COLOR_DRONE, linewidth=1.6, label="导弹—云团中心 距离")
    ax.axhline(cfg.EFFECTIVE_RADIUS, color=COLOR_ACCENT, linewidth=1.0, linestyle=":",
               label=f"有效遮蔽半径 {cfg.EFFECTIVE_RADIUS:.0f} m")
    ax.axvline(detonation_time, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(detonation_time, ax.get_ylim()[1] * 0.92, f" 起爆 t={detonation_time:.1f}s",
            fontsize=8, va="top")

    ax.set_xlabel("t (s)")
    ax.set_ylabel("距离 (m)")
    ax.set_title(
        f"问题1场景：release={release_time}s, delay={delay}s "
        f"(下沉速度{cfg.SMOKE_SINK_SPEED}m/s, 速度范围{cfg.DRONE_SPEED_MIN:.0f}-{cfg.DRONE_SPEED_MAX:.0f}m/s)",
        fontsize=10,
    )
    ax.legend(loc="center right", frameon=False, fontsize=9)

    # 放大视图：主图的纵轴尺度(万米级)看不出10m阈值附近的细节，
    # 单开一个内嵌坐标轴聚焦到最近距离出现的那一小段时间窗口
    axins = ax.inset_axes([0.10, 0.14, 0.30, 0.42])
    win = (ts > t_min - 1.5) & (ts < t_min + 1.5)
    axins.plot(ts[win], dist_ms[win], color=COLOR_DRONE, linewidth=1.4)
    axins.axhline(cfg.EFFECTIVE_RADIUS, color=COLOR_ACCENT, linewidth=1.0, linestyle=":")
    axins.scatter([t_min], [d_min], color=COLOR_DRONE, s=18, zorder=5)
    axins.annotate(f"最近 {d_min:.2f} m\n(阈值{cfg.EFFECTIVE_RADIUS:.0f}m，差{d_min-cfg.EFFECTIVE_RADIUS:.2f}m)",
                    xy=(t_min, d_min), xytext=(0.55, 0.75), textcoords="axes fraction",
                    fontsize=7.5, ha="left",
                    arrowprops=dict(arrowstyle="-", color="#666666", linewidth=0.6))
    axins.set_xlabel("t (s)", fontsize=7)
    axins.tick_params(labelsize=7)
    axins.grid(alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    return fig, ax


def plot_pso_convergence(history, title="PSO收敛曲线", maximize=True, label=None):
    """
    PSO收敛曲线：history 是 PSO.optimize() 跑完之后 pso.history 那个list
    (内部存的是"按求最小值"语义的值，maximize=True时要取负号还原成真实目标值)
    """
    history = np.asarray(history, dtype=float)
    values = -history if maximize else history

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(values)), values, color=COLOR_DRONE, linewidth=1.4)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("最优目标值 (s)")
    ax.set_title(title, fontsize=11)
    if label:
        ax.text(0.98, 0.05, label, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#666666")
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import os
    outdir = os.path.join(os.path.dirname(__file__), "..", "paper_assets", "generated")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    fig1, _ = plot_scenario_setup()
    p1 = os.path.join(outdir, "场景初始布局_v1.png")
    fig1.savefig(p1, dpi=200)
    print("saved:", p1)

    fig2, _ = plot_problem1_timeline()
    p2 = os.path.join(outdir, "问题1距离时序_v1.png")
    fig2.savefig(p2, dpi=200)
    print("saved:", p2)
