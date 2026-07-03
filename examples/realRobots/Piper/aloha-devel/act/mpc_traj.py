#!/usr/bin/env python
# -*- coding: utf-8 -*-
## 其他常用库
import numpy as np
import time
import signal
import sys
import select
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#import gripper0 as grp

import rospy
import move_piper as dmmv
import move_piper_left as dmmvl
import mpc_piper as dmpc
import mpc_piper_single as smpc
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
left_actual_angles = []
right_actual_angles = []

# 定义目标角度
q_goal = np.array([0.16, -0.13, 2.22, -2.79, 0.97, -0.22, 1.21, -4.06, 0.30, -0.31, -0.45, 1.75])
q_init = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
left_goal_angles = q_goal[:6]  # 左臂目标角度
right_goal_angles = q_goal[6:]  # 右臂目标角度
left_init_angles = q_init[:6]  # 左臂初始角度
right_init_angles = q_init[6:]  # 右臂初始角度

# 监听退出信号
def signal_handler(sig, frame):
    print('\n程序终止中...')
    sys.exit(0)

# 捕获 Ctrl+C 中断信号
signal.signal(signal.SIGINT, signal_handler)

left_joint_angles = []
right_joint_angles = []

left_joint_angles.append(left_init_angles)
right_joint_angles.append(right_init_angles)
left_joint_angles.append(left_goal_angles)
right_joint_angles.append(right_goal_angles)


def fill_missing_points(joint_angles, start_index, end_index, num_points=100):
    """
    填补路径点之间的缺口
    :param joint_angles: 原始路径点列表
    :param start_index: 缺口的起始索引
    :param end_index: 缺口的结束索引
    :param num_points: 插值点的数量
    :return: 填补后的路径点列表
    """
    start = np.array(joint_angles[start_index])
    end = np.array(joint_angles[end_index])
    interpolated_points = [
        (1 - t) * start + t * end for t in np.linspace(0, 1, num_points, endpoint=False)
    ]
    return joint_angles[:start_index + 1] + interpolated_points + joint_angles[end_index:]
    


# 确保 end_index 不超过列表长度
end_index_left = min(400, len(left_joint_angles) - 1)
end_index_right = min(400, len(right_joint_angles) - 1)

# 填补缺口
left_joint_angles = fill_missing_points(left_joint_angles, 0, end_index_left, num_points=100)
right_joint_angles = fill_missing_points(right_joint_angles, 0, end_index_right, num_points=100)


dmmv.move_arms(
        left_joint_positions=left_init_angles,
        right_joint_positions=right_init_angles
        )
print('机械臂到达初始位置')

m = len(left_joint_angles)
state = [0,0,0,0,0,0,0,0,0]
x_init1 = np.hstack([state, left_init_angles]).reshape(-1, 1)
x_init2 = np.hstack([state, right_init_angles]).reshape(-1, 1)
mpc = 1
start_time = time.time()

for i in range(m):
    try:
        if mpc == 1:
            #state1 = dmpc1.fkine_ur5e(left_joint_angles[1])
            #state2 = dmpc1.fkine_ur3(right_joint_angles[i])
            
            xs1 = np.hstack([state, left_joint_angles[i]])
            xs1 = xs1.reshape(-1, 1)
            xs2 = np.hstack([state, right_joint_angles[i]])
            xs2 = xs2.reshape(-1, 1)
            is_mpc = dmpc.ur_move(x_init1, xs1, x_init2, xs2)
            if is_mpc == 1:
                x_init1 = xs1
                x_init2 = xs2
            else:
                continue
        else:
            dmmv.move_arms(left_joint_angles[i], right_joint_angles[i])
        left_actual_angles.append(xs1[9:15].flatten())
        right_actual_angles.append(xs2[9:15].flatten())


        print('已执行', i + 1, '步,共', m, '步')
        #print('ur5e当前位置:', xs1, '\n')
        #print('ur3当前位置:', xs2, '\n')
    except KeyboardInterrupt:
        print("\n程序被手动中止")
        break

# 转换为 NumPy 数组
left_actual_angles = np.array(left_actual_angles)
right_actual_angles = np.array(right_actual_angles)

rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用 Noto Sans 字体（适用于 Ubuntu）
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

print(  '左臂实际角度:', left_actual_angles[-1]) 
print(  '右臂实际角度:', right_actual_angles[-1])   
print(  '左臂目标角度:', left_goal_angles)
print(  '右臂目标角度:', right_goal_angles)         
# 计算最终误差
left_final_error = left_goal_angles - left_actual_angles[-1]
right_final_error = right_goal_angles - right_actual_angles[-1]
print('左臂最终误差:', left_final_error)
print('右臂最终误差:', right_final_error)
end_time = time.time()
t = end_time - start_time
averge = t/m
print('总耗时:',t,'s')
print('平均每步耗时:',averge,'s')


# 左臂所有关节在一个图
plt.figure(figsize=(10, 6))
for joint in range(6):
    plt.plot(left_actual_angles[:, joint], label=f'左臂关节 {joint + 1} 实际角度')
    plt.axhline(y=left_goal_angles[joint], linestyle='--', 
                label=f'左臂关节 {joint + 1} 目标角度')
    # 标注误差
    plt.text(len(left_actual_angles) - 1, left_goal_angles[joint], 
             f'误差: {left_final_error[joint]:.4f}', 
             fontsize=8, ha='right')

plt.title('左臂 6 个关节角度跟踪', fontsize=14)
plt.xlabel('步数', fontsize=12)
plt.ylabel('角度 (弧度)', fontsize=12)
plt.legend(fontsize=8, ncol=2)  # 图例排成两列
plt.grid()
plt.tight_layout()
plt.savefig('left_arm_tracking.png')
plt.show()


# 右臂所有关节在一个图
plt.figure(figsize=(10, 6))
for joint in range(6):
    plt.plot(right_actual_angles[:, joint], label=f'右臂关节 {joint + 1} 实际角度')
    plt.axhline(y=right_goal_angles[joint], linestyle='--', 
                label=f'右臂关节 {joint + 1} 目标角度')
    plt.text(len(right_actual_angles) - 1, right_goal_angles[joint], 
             f'误差: {right_final_error[joint]:.4f}', 
             fontsize=8, ha='right')

plt.title('右臂 6 个关节角度跟踪', fontsize=14)
plt.xlabel('步数', fontsize=12)
plt.ylabel('角度 (弧度)', fontsize=12)
plt.legend(fontsize=8, ncol=2)
plt.grid()
plt.tight_layout()
plt.savefig('right_arm_tracking.png')
plt.show()



