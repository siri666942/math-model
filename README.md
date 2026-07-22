# C题 烟幕干扰弹的投放策略 — 项目文档

## 项目结构

```
数学建模国赛/
├── C题.pdf                              # 题目原文 (moni2026 C题)
├── 全国大学生数学建模竞赛A题.pdf          # 参考解法 (2025真题A题)
├── C题改动报告.md                        # 参数差异对比
├── 支撑材料/                             # 原始MATLAB代码（A题解法）
│   ├── init.m          # 3D可视化
│   ├── m_1_1.m         # 问题1 (M1 + FY1)
│   ├── m_2.m           # 问题2 (PSO优化)
│   ├── m3.m            # 问题3 (FY1三弹)
│   ├── m_4.m           # 问题4 (三机协同)
│   ├── m_5.m           # 问题5 (五机多弹)
│   ├── Obj_fun.m       # 目标函数
│   ├── cal_xyz.m       # 坐标反推
│   ├── minggandu.m     # 敏感性分析
│   └── fig.m / fy.m    # 可视化
│
└── py_solution/                          # Python求解（本项目）
    ├── README.md        # ★ 详细文档
    ├── config.py        # C题参数配置
    ├── simulation.py    # 核心仿真引擎
    ├── pso.py           # PSO优化器
    ├── final_solve.py   # ★ 主求解入口
    ├── problem1.py      # 问题1
    ├── problem2.py      # 问题2
    ├── problem3.py      # 问题3
    ├── problem4.py      # 问题4
    ├── problem5.py      # 问题5
    ├── result1.xlsx     # 问题3输出
    ├── result2.xlsx     # 问题4输出
    └── result3.xlsx     # 问题5输出
```

## 快速开始

```bash
cd py_solution
pip install numpy scipy openpyxl
python -u final_solve.py
```

## 最终结果一览

| 问题 | C题结果 | A题参考 |
|------|---------|---------|
| 1 | **0.0000 s** | 1.3915 s |
| 2 | **4.8200 s** | 4.5960 s |
| 3 | **4.9000 s** | 7.6500 s |
| 4 | **8.3050 s** | 11.7540 s |
| 5 | **5.8350 s** | 38.0600 s |

## 详细文档

参见 [py_solution/README.md](py_solution/README.md)
