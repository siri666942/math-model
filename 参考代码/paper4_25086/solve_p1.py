"""
问题1: 利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰
给定飞行参数，计算有效遮蔽时长

Paper cumcm25086 approach: direct simulation with fixed parameters
Uses refined geometric occlusion model + vectorized computation
"""
import numpy as np
import simulation as sim


def solve_p1(config_module, dt_fine=None):
    """
    求解问题1

    Parameters:
        config_module: config_a or config_c module
        dt_fine: fine time step for accurate computation (default config's DT_FINE)

    Returns:
        effective_time: total effective coverage time (seconds)
    """
    sim.set_config(config_module)

    if dt_fine is None:
        dt_fine = config_module.DT_FINE

    drone_init = config_module.DRONES_INIT[0]  # FY1
    theta = config_module.P1_DRONE_THETA        # pi (toward fake target)
    speed = config_module.P1_DRONE_SPEED        # 120 m/s
    release_time = config_module.P1_RELEASE_TIME
    detonation_delay = config_module.P1_DETONATION_DELAY

    print("=" * 60)
    print("问题1: FY1投放1枚烟幕干扰弹对M1的有效遮蔽时长")
    print("=" * 60)
    print(f"\n参数:")
    print(f"  烟雾下沉速度: {config_module.SMOKE_SINK_SPEED} m/s")
    print(f"  无人机速度范围: [{config_module.DRONE_SPEED_MIN}, {config_module.DRONE_SPEED_MAX}] m/s")
    print(f"  FY1初始位置: ({drone_init[0]:.1f}, {drone_init[1]:.1f}, {drone_init[2]:.1f})")
    print(f"  航向角theta: {theta:.4f} rad ({np.degrees(theta):.2f} deg)")
    print(f"  飞行速度: {speed} m/s")
    print(f"  投放时间: {release_time} s")
    print(f"  起爆延时: {detonation_delay} s")

    target_keypoints = sim.get_target_keypoints()

    effective_time = sim.simulate_single_bomb(
        drone_init, theta, speed, release_time, detonation_delay,
        missile_idx=0, target_keypoints=target_keypoints,
        dt=dt_fine, t_total=config_module.T_TOTAL
    )

    print(f"\n结果:")
    print(f"  对M1的有效遮蔽时长: {effective_time:.4f} s")

    # 计算关键点
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    release_pos = drone_init + speed * direction * release_time
    detonation_pos = release_pos + speed * direction * detonation_delay
    detonation_pos[2] -= 0.5 * config_module.G * detonation_delay ** 2
    detonation_time = release_time + detonation_delay

    print(f"  投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
    print(f"  起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")
    print(f"  起爆时间: {detonation_time:.4f} s")
    print(f"  烟云有效截止时间: {detonation_time + config_module.EFFECTIVE_DURATION:.4f} s")

    return effective_time


if __name__ == "__main__":
    import config_a
    solve_p1(config_a)
