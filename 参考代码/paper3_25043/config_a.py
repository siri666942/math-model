"""
config_a.py - A题 original parameters
Based on cumcm25043 paper, 2025 CUMCM Problem A
"""
import numpy as np

G = 9.8
SMOKE_SINK_SPEED = 3.0          # 下沉速度: 3 m/s
EFFECTIVE_RADIUS = 10.0         # 有效烟幕半径
EFFECTIVE_DURATION = 20.0       # 有效持续时间
MISSILE_SPEED = 300.0           # 导弹速度
DRONE_SPEED_MIN = 70.0          # 无人机最小飞行速度
DRONE_SPEED_MAX = 140.0         # 无人机最大飞行速度
BOMB_INTERVAL_MIN = 1.0         # 最小投弹间隔
P1_RELEASE_TIME = 1.5           # 问题1投放时间
P1_DETONATION_DELAY = 3.6       # 问题1起爆延时
P1_DRONE_SPEED = 120.0          # 问题1无人机速度
P1_DRONE_THETA = np.pi          # 问题1无人机进入角度(0=沿X轴; pi=沿-X轴)
TARGET_CENTER = np.array([0.0, 200.0, 0.0])     # 真目标中心
TARGET_RADIUS = 7.0             # 目标半径
TARGET_HEIGHT = 10.0            # 目标高度
FAKE_TARGET = np.array([0.0, 0.0, 0.0])         # 假目标
# 5枚导弹初始位置
MISSILES_INIT = np.array([
    [20000.0, 0.0, 2000.0],
    [19000.0, 600.0, 2100.0],
    [18000.0, -600.0, 1900.0],
    [17000.0, 400.0, 2000.0],
    [16000.0, -400.0, 1950.0]
])
# 5架无人机初始位置 (P5 only, but define all)
DRONES_INIT = np.array([
    [17800.0, 0.0, 1800.0],
    [12000.0, 1400.0, 1400.0],
    [6000.0, -3000.0, 700.0],
    [11000.0, 2000.0, 1800.0],
    [13000.0, -2000.0, 1300.0]
])
DT = 0.005
DT_FINE = 0.0001
T_TOTAL = 50.0
# Paper uses projection-based method, fewer keypoints
N_CIRCLE_POINTS = 36          # 水平方向采样点
N_SIDE_LAYERS = 5             # 垂直方向层数
# PSO parameters from paper
W_START = 0.9
W_END = 0.4
C1 = 1.49
C2 = 1.49
