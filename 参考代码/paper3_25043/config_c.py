"""
config_c.py - C题 modified parameters
SMOKE_SINK_SPEED = 2.5, DRONE_SPEED_MIN = 80, DRONE_SPEED_MAX = 120
"""
import numpy as np

G = 9.8
SMOKE_SINK_SPEED = 2.5          # C题: 2.5 m/s
EFFECTIVE_RADIUS = 10.0
EFFECTIVE_DURATION = 20.0
MISSILE_SPEED = 300.0
DRONE_SPEED_MIN = 80.0          # C题: 80 m/s
DRONE_SPEED_MAX = 120.0         # C题: 120 m/s
BOMB_INTERVAL_MIN = 1.0
P1_RELEASE_TIME = 1.2           # C题: 1.2s
P1_DETONATION_DELAY = 3.2       # C题: 3.2s
P1_DRONE_SPEED = 120.0
P1_DRONE_THETA = np.pi
TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
MISSILES_INIT = np.array([
    [20000.0, 0.0, 2000.0],
    [19000.0, 600.0, 2100.0],
    [18000.0, -600.0, 1900.0],
    [17000.0, 400.0, 2000.0],
    [16000.0, -400.0, 1950.0]
])
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
N_CIRCLE_POINTS = 36
N_SIDE_LAYERS = 5
# PSO parameters from paper
W_START = 0.9
W_END = 0.4
C1 = 1.49
C2 = 1.49
