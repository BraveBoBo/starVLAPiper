#!/usr/bin/env python
# -*- coding: utf-8 -*-
## CasADi 工具
import casadi as ca
## 其他常用库
import numpy as np
import time
# import matplotlib.pyplot as plt
import math
# from mpl_toolkits.mplot3d import Axes3D
#import gripper0 as grp
from pynput import keyboard
import sys

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import rospy

class MPC_piper:
    def __init__(self,pub):
        self.action_pub = pub

    def init_solver(self,n_states,n_controls,N,f,T):
        # 开始构建MPC
        ## 相关变量，格式(状态长度， 步长)
        U = ca.SX.sym('U', n_controls, N) # N步内的控制输出
        X = ca.SX.sym('X', n_states, N+1) # N+1步的系统状态，通常长度比控制多1
        P = ca.SX.sym('P', n_states+n_states) # 构建问题的相关参数
                                              # 在这里每次只需要给定当前/初始位置和目标终点位置
        ## Single Shooting 约束条件
        X[:, 0] = P[:15] # 初始状态希望相等

        ### 剩余N状态约束条件
        for i in range(N):
            # 通过前述函数获得下个时刻系统状态变化。
            # 这里需要注意引用的index为[:, i]，因为X为(n_states, N+1)矩阵 
            f_value = f(X[:, i], U[:, i]) 
            X[:, i+1] = X[:, i] + f_value*T
        ## 获得输入（控制输入，参数）和输出（系统状态）之间关系的函数ff
        ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])



        ## 
        # ### 惩罚矩阵
        #
        Q = np.diag([2,2,2,0.5,0.5,0.5,0.5,0.5,0.5])*1 #
        Q_joint = np.diag([0.1,0.1,0.1,0.1,0.1,0.1])*0 # 关节角度误差惩罚
        R = np.diag([0.01,0.01,0.01,0.01,0.01,0.01])*1 #6*6*0
        # R = R0
        ### 优化目标
        obj = 0 # 初始化优化目标值
        for i in range(N):
            # 在N步内对获得优化目标表达式
            obj = obj + ca.mtimes([(X[:9, i]-P[15:24]).T, Q, X[:9, i]-P[15:24]]) + ca.mtimes([U[:, i].T, R, U[:, i]])+ca.mtimes([(X[9:15, i]-P[24:30]).T, Q_joint, X[9:15, i]-P[24:30]])
        
        ### 约束条件定义0
        g = [] # 用list来存储优化目标的向量
        for i in range(N+1):
            g.append(X[9, i])
            g.append(X[10, i])
            g.append(X[11, i])
            g.append(X[12, i])
            g.append(X[13, i])
            g.append(X[14, i])


        ### 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
        ### 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
        nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
        ### ipot设置
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        ## 最终目标，获得求解器
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        return ff,solver

    def fkine_piper(self,theta):
        theta1,theta2,theta3,theta4,theta5,theta6 = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5]

        x0 = -0.14*((6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.sin(theta5) - np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4))*np.cos(theta6) \
            - 0.091*(6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.cos(theta5) \
            - 0.14*(6.12323399573677e-17*np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + 3.77373430684139e-49*np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4) + 6.12323399573677e-17*np.sin(theta1 + theta2)*np.cos(theta5) - 6.12323399573677e-17*np.sin(theta1 + theta2) + 3.74939945665464e-33*np.sin(theta3 + theta4)*np.cos(theta5)*np.cos(theta1 + theta2) + np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.sin(theta6) \
            - 0.25075*np.sin(theta1)*np.cos(theta2) - 0.25075*np.sin(theta2)*np.cos(theta1) - 1.74530538580485e-17*np.sin(theta3)*np.sin(theta1 + theta2) - 0.091*np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 2.78607146806023e-18*np.sin(theta1 + theta2 - theta3 - theta4) + 2.78607146806023e-18*np.sin(theta1 + theta2 + theta3 + theta4) + 0.28503*np.cos(theta3)*np.cos(theta1 + theta2) - 0.01099*np.cos(theta1 + theta2 - theta3 - theta4) - 0.01099*np.cos(theta1 + theta2 + theta3 + theta4)

        y0 = 0.14*((-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.sin(theta5) + np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4))*np.cos(theta6) \
            + 0.091*(-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.cos(theta5) \
            - 0.14*(6.12323399573677e-17*np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.sin(theta1 + theta2)*np.sin(theta3 + theta4)*np.cos(theta5) + np.sin(theta1 + theta2)*np.sin(theta3 + theta4) - 3.77373430684139e-49*np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 6.12323399573677e-17*np.cos(theta5)*np.cos(theta1 + theta2) + 6.12323399573677e-17*np.cos(theta1 + theta2))*np.sin(theta6) \
            - 0.25075*np.sin(theta1)*np.sin(theta2) + 1.74530538580485e-17*np.sin(theta3)*np.cos(theta1 + theta2) - 0.091*np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 0.28503*np.sin(theta1 + theta2)*np.cos(theta3) - 0.01099*np.sin(theta1 + theta2 - theta3 - theta4) - 0.01099*np.sin(theta1 + theta2 + theta3 + theta4) + 0.25075*np.cos(theta1)*np.cos(theta2) + 2.78607146806023e-18*np.cos(theta1 + theta2 - theta3 - theta4) - 2.78607146806023e-18*np.cos(theta1 + theta2 + theta3 + theta4)

        z0 = -0.28503*np.sin(theta3) + 0.091*np.sin(theta5)*np.sin(theta3 + theta4) + 8.57252759403147e-18*np.sin(theta5)*np.cos(theta6) - 8.57252759403147e-18*np.sin(theta5)*np.cos(theta3 + theta4 + theta6) - 5.2491592393165e-34*np.sin(theta6)*np.cos(theta5)*np.cos(theta3 + theta4) + 5.2491592393165e-34*np.sin(theta6)*np.cos(theta5) - 0.14*np.sin(theta6)*np.cos(theta3 + theta4) - 5.2491592393165e-34*np.sin(theta6) - 0.14*np.sin(theta3 + theta4)*np.cos(theta5)*np.cos(theta6) + 0.02198*np.sin(theta3 + theta4) - 5.57214293612046e-18*np.cos(theta5)*np.cos(theta3 + theta4) + 5.57214293612046e-18*np.cos(theta5) + 5.57214293612046e-18*np.cos(theta3 + theta4) + 0.123

        nx = -((6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.sin(theta5) - np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4))*np.cos(theta6) - (6.12323399573677e-17*np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + 3.77373430684139e-49*np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4) + 6.12323399573677e-17*np.sin(theta1 + theta2)*np.cos(theta5) - 6.12323399573677e-17*np.sin(theta1 + theta2) + 3.74939945665464e-33*np.sin(theta3 + theta4)*np.cos(theta5)*np.cos(theta1 + theta2) + np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.sin(theta6)

        ny = ((-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.sin(theta5) + np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4))*np.cos(theta6) - (6.12323399573677e-17*np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.sin(theta1 + theta2)*np.sin(theta3 + theta4)*np.cos(theta5) + np.sin(theta1 + theta2)*np.sin(theta3 + theta4) - 3.77373430684139e-49*np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 6.12323399573677e-17*np.cos(theta5)*np.cos(theta1 + theta2) + 6.12323399573677e-17*np.cos(theta1 + theta2))*np.sin(theta6)

        nz = -(6.12323399573677e-17*(np.cos(theta3 + theta4) - 1)*np.sin(theta5) + np.sin(theta3 + theta4)*np.cos(theta5))*np.cos(theta6) - (-6.12323399573677e-17*np.sin(theta5)*np.sin(theta3 + theta4) + 3.74939945665464e-33*np.cos(theta5)*np.cos(theta3 + theta4) - 3.74939945665464e-33*np.cos(theta5) + np.cos(theta3 + theta4) + 3.74939945665464e-33)*np.sin(theta6)

        ox = 6.12323399573677e-17*((6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.sin(theta5) - np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4))*np.sin(theta6) - (6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.cos(theta5) - 6.12323399573677e-17*(6.12323399573677e-17*np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + 3.77373430684139e-49*np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4) + 6.12323399573677e-17*np.sin(theta1 + theta2)*np.cos(theta5) - 6.12323399573677e-17*np.sin(theta1 + theta2) + 3.74939945665464e-33*np.sin(theta3 + theta4)*np.cos(theta5)*np.cos(theta1 + theta2) + np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.cos(theta6) - 3.74939945665464e-33*np.sin(theta1)*np.cos(theta2) - 3.74939945665464e-33*np.sin(theta2)*np.cos(theta1) - np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 3.06161699786838e-17*np.sin(theta1 + theta2 - theta3 - theta4) + 3.06161699786838e-17*np.sin(theta1 + theta2 + theta3 + theta4)

        oy = -6.12323399573677e-17*((-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.sin(theta5) + np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4))*np.sin(theta6) + (-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.cos(theta5) - 6.12323399573677e-17*(6.12323399573677e-17*np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.sin(theta1 + theta2)*np.sin(theta3 + theta4)*np.cos(theta5) + np.sin(theta1 + theta2)*np.sin(theta3 + theta4) - 3.77373430684139e-49*np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 6.12323399573677e-17*np.cos(theta5)*np.cos(theta1 + theta2) + 6.12323399573677e-17*np.cos(theta1 + theta2))*np.cos(theta6) - 3.74939945665464e-33*np.sin(theta1)*np.sin(theta2) - np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.cos(theta1)*np.cos(theta2) + 3.06161699786838e-17*np.cos(theta1 + theta2 - theta3 - theta4) - 3.06161699786838e-17*np.cos(theta1 + theta2 + theta3 + theta4)

        oz = 6.12323399573677e-17*(1 - np.cos(theta3 + theta4))*np.cos(theta5) + 6.12323399573677e-17*(-6.12323399573677e-17*(1 - np.cos(theta3 + theta4))*np.sin(theta5) + np.sin(theta3 + theta4)*np.cos(theta5))*np.sin(theta6) + 6.12323399573677e-17*(6.12323399573677e-17*np.sin(theta5)*np.sin(theta3 + theta4) - 3.74939945665464e-33*np.cos(theta5)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.cos(theta5) - np.cos(theta3 + theta4) - 3.74939945665464e-33)*np.cos(theta6) - 6.12323399573677e-17*np.sin(theta3)*np.sin(theta4) + np.sin(theta5)*np.sin(theta3 + theta4) + 6.12323399573677e-17*np.cos(theta3)*np.cos(theta4) + 2.29584502165847e-49

        ax = -((6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.sin(theta5) - np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4))*np.sin(theta6) - 6.12323399573677e-17*(6.16297582203915e-33*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + np.sin(theta1 + theta2) + 6.12323399573676e-17*np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.cos(theta5) + (6.12323399573677e-17*np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + 3.77373430684139e-49*np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4) + 6.12323399573677e-17*np.sin(theta1 + theta2)*np.cos(theta5) - 6.12323399573677e-17*np.sin(theta1 + theta2) + 3.74939945665464e-33*np.sin(theta3 + theta4)*np.cos(theta5)*np.cos(theta1 + theta2) + np.sin(theta3 + theta4)*np.cos(theta1 + theta2))*np.cos(theta6) - 2.29584502165847e-49*np.sin(theta1)*np.cos(theta2) - 2.29584502165847e-49*np.sin(theta2)*np.cos(theta1) - 6.12323399573677e-17*np.sin(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 1.87469972832732e-33*np.sin(theta1 + theta2 - theta3 - theta4) + 1.87469972832732e-33*np.sin(theta1 + theta2 + theta3 + theta4)

        ay = ((-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.sin(theta5) + np.sin(theta1 + theta2)*np.cos(theta5)*np.cos(theta3 + theta4))*np.sin(theta6) + 6.12323399573677e-17*(-6.12323399573676e-17*np.sin(theta1 + theta2)*np.sin(theta3 + theta4) + 6.16297582203915e-33*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) + np.cos(theta1 + theta2))*np.cos(theta5) + (6.12323399573677e-17*np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.sin(theta1 + theta2)*np.sin(theta3 + theta4)*np.cos(theta5) + np.sin(theta1 + theta2)*np.sin(theta3 + theta4) - 3.77373430684139e-49*np.cos(theta5)*np.cos(theta1 + theta2)*np.cos(theta3 + theta4) - 6.12323399573677e-17*np.cos(theta5)*np.cos(theta1 + theta2) + 6.12323399573677e-17*np.cos(theta1 + theta2))*np.cos(theta6) - 2.29584502165847e-49*np.sin(theta1)*np.sin(theta2) - 6.12323399573677e-17*np.sin(theta5)*np.sin(theta1 + theta2)*np.cos(theta3 + theta4) + 2.29584502165847e-49*np.cos(theta1)*np.cos(theta2) + 1.87469972832732e-33*np.cos(theta1 + theta2 - theta3 - theta4) - 1.87469972832732e-33*np.cos(theta1 + theta2 + theta3 + theta4)

        az = 3.74939945665464e-33*(1 - np.cos(theta3 + theta4))*np.cos(theta5) - ((-6.12323399573677e-17*(1 - np.cos(theta3 + theta4))*np.sin(theta5) + np.sin(theta3 + theta4)*np.cos(theta5))*np.sin(theta6)) - ((6.12323399573677e-17*np.sin(theta5)*np.sin(theta3 + theta4) - 3.74939945665464e-33*np.cos(theta5)*np.cos(theta3 + theta4) + 3.74939945665464e-33*np.cos(theta5) - np.cos(theta3 + theta4) - 3.74939945665464e-33)*np.cos(theta6)) - 3.74939945665464e-33*np.sin(theta3)*np.sin(theta4) + 6.12323399573677e-17*np.sin(theta5)*np.sin(theta3 + theta4) + 3.74939945665464e-33*np.cos(theta3)*np.cos(theta4) + 1.40579962855621e-65

        state0=[x0,y0,z0,ox,oy,oz,ax,ay,az]
        return state0



    def init_solver(self,n_states,n_controls,N,f,T):
        # 开始构建MPC
        ## 相关变量，格式(状态长度， 步长)
        U = ca.SX.sym('U', n_controls, N) # N步内的控制输出
        X = ca.SX.sym('X', n_states, N+1) # N+1步的系统状态，通常长度比控制多1
        P = ca.SX.sym('P', n_states+n_states) # 构建问题的相关参数
                                            # 在这里每次只需要给定当前/初始位置和目标终点位置
        ## Single Shooting 约束条件
        X[:, 0] = P[:15] # 初始状态希望相等

        ### 剩余N状态约束条件
        for i in range(N):
            # 通过前述函数获得下个时刻系统状态变化。
            # 这里需要注意引用的index为[:, i]，因为X为(n_states, N+1)矩阵 
            f_value = f(X[:, i], U[:, i]) 
            X[:, i+1] = X[:, i] + f_value*T
        ## 获得输入（控制输入，参数）和输出（系统状态）之间关系的函数ff
        ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])



        ## 
        # ### 惩罚矩阵
        #
        Q = np.diag([2,2,2,0.5,0.5,0.5,0.5,0.5,0.5])*1 #
        Q_joint = np.diag([0.1,0.1,0.1,0.1,0.1,0.1])*0 # 关节角度误差惩罚
        R = np.diag([0.01,0.01,0.01,0.01,0.01,0.01])*1 #6*6*0
        # R = R0
        ### 优化目标
        obj = 0 # 初始化优化目标值
        for i in range(N):
            # 在N步内对获得优化目标表达式
            obj = obj + ca.mtimes([(X[:9, i]-P[15:24]).T, Q, X[:9, i]-P[15:24]]) + ca.mtimes([U[:, i].T, R, U[:, i]])+ca.mtimes([(X[9:15, i]-P[24:30]).T, Q_joint, X[9:15, i]-P[24:30]])
        
        ### 约束条件定义0
        g = [] # 用list来存储优化目标的向量
        for i in range(N+1):
            g.append(X[9, i])
            g.append(X[10, i])
            g.append(X[11, i])
            g.append(X[12, i])
            g.append(X[13, i])
            g.append(X[14, i])


        ### 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
        ### 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
        nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
        ### ipot设置
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        ## 最终目标，获得求解器
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        return ff,solver

    def shift_movement(self,T, t0, x0, u, f):
        # 
        f_value = f(x0, u[:, 0])
        st_0 = x0 + T*f_value
        st0 = [st_0[9],st_0[10],st_0[11],st_0[12],st_0[13],st_0[14]]
        # 时间增加
        st1 = self.fkine_piper(st0)
        st = np.concatenate([st1,st0])
        t = t0 + T
        # 准备下一个估计的最优控制，因为u[:, 0]已经采纳，我们就简单地把后面的结果提前
        u_end = ca.horzcat(u[:, 1:], u[:, -1])
        return t, st, u_end.T
    
    def move_arms(self,left_joint_positions,grab):
        pos = np.append(left_joint_positions, grab)

        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = pos

        # rospy.loginfo(joint_state_msg)

        self.action_pub.publish(joint_state_msg)
        


    def ur_move(self,x_init,xs,grab):
    # if __name__ == "__main__":
        T = 0.02 # （模拟的）系统采样时间【秒】
        N = 3 # 需要预测的步长【超参数】

        omega_max = [2,2,2,2,2,2]
        omega_max_1 = [-2,-2,-2,-2,-2,-2]
        # omega_max = [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf]
        # omega_max_1 = [-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf] # 最大转动角速度 【物理约束】
    
        # 根据数学模型建模
        ## 系统状态    a_x = ca.SX.sym('a_x')
        a_y = ca.SX.sym('a_y')
        a_z = ca.SX.sym('a_z')
        theta1 = ca.SX.sym('theta1')
        theta2 = ca.SX.sym('theta2')
        theta3 = ca.SX.sym('theta3')
        theta4 = ca.SX.sym('theta4')
        theta5 = ca.SX.sym('theta5')
        theta6 = ca.SX.sym('theta6')
        theta = ca.vertcat(theta1, theta2, theta3, theta4, theta5, theta6)


        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        z = ca.SX.sym('z')
        a_x = ca.SX.sym('a_x')
        a_y = ca.SX.sym('a_y')
        a_z = ca.SX.sym('a_z')
        o_x = ca.SX.sym('o_x')
        o_y = ca.SX.sym('o_y')
        o_z = ca.SX.sym('o_z')


        #leftpiper
        x0 = -0.14*((6.16297582203915e-33*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.sin(theta1 + theta2) + 6.12323399573676e-17*ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.sin(theta5) - ca.cos(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4))*ca.cos(theta6) \
            - 0.091*(6.16297582203915e-33*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.sin(theta1 + theta2) + 6.12323399573676e-17*ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.cos(theta5) \
            - 0.14*(6.12323399573677e-17*ca.sin(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.77373430684139e-49*ca.sin(theta1 + theta2)*ca.cos(theta5)*ca.cos(theta3 + theta4) + 6.12323399573677e-17*ca.sin(theta1 + theta2)*ca.cos(theta5) - 6.12323399573677e-17*ca.sin(theta1 + theta2) + 3.74939945665464e-33*ca.sin(theta3 + theta4)*ca.cos(theta5)*ca.cos(theta1 + theta2) + ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.sin(theta6) \
            - 0.25075*ca.sin(theta1)*ca.cos(theta2) - 0.25075*ca.sin(theta2)*ca.cos(theta1) - 1.74530538580485e-17*ca.sin(theta3)*ca.sin(theta1 + theta2) - 0.091*ca.sin(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) - 2.78607146806023e-18*ca.sin(theta1 + theta2 - theta3 - theta4) + 2.78607146806023e-18*ca.sin(theta1 + theta2 + theta3 + theta4) + 0.28503*ca.cos(theta3)*ca.cos(theta1 + theta2) - 0.01099*ca.cos(theta1 + theta2 - theta3 - theta4) - 0.01099*ca.cos(theta1 + theta2 + theta3 + theta4)

        y0 = 0.14*((-6.12323399573676e-17*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) + 6.16297582203915e-33*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.cos(theta1 + theta2))*ca.sin(theta5) + ca.sin(theta1 + theta2)*ca.cos(theta5)*ca.cos(theta3 + theta4))*ca.cos(theta6) \
            + 0.091*(-6.12323399573676e-17*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) + 6.16297582203915e-33*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.cos(theta1 + theta2))*ca.cos(theta5) \
            - 0.14*(6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.74939945665464e-33*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4)*ca.cos(theta5) + ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) - 3.77373430684139e-49*ca.cos(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) - 6.12323399573677e-17*ca.cos(theta5)*ca.cos(theta1 + theta2) + 6.12323399573677e-17*ca.cos(theta1 + theta2))*ca.sin(theta6) \
            - 0.25075*ca.sin(theta1)*ca.sin(theta2) + 1.74530538580485e-17*ca.sin(theta3)*ca.cos(theta1 + theta2) - 0.091*ca.sin(theta5)*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + 0.28503*ca.sin(theta1 + theta2)*ca.cos(theta3) - 0.01099*ca.sin(theta1 + theta2 - theta3 - theta4) - 0.01099*ca.sin(theta1 + theta2 + theta3 + theta4) + 0.25075*ca.cos(theta1)*ca.cos(theta2) + 2.78607146806023e-18*ca.cos(theta1 + theta2 - theta3 - theta4) - 2.78607146806023e-18*ca.cos(theta1 + theta2 + theta3 + theta4)

        z0 = -0.28503*ca.sin(theta3) + 0.091*ca.sin(theta5)*ca.sin(theta3 + theta4) + 8.57252759403147e-18*ca.sin(theta5)*ca.cos(theta6) - 8.57252759403147e-18*ca.sin(theta5)*ca.cos(theta3 + theta4 + theta6) - 5.2491592393165e-34*ca.sin(theta6)*ca.cos(theta5)*ca.cos(theta3 + theta4) + 5.2491592393165e-34*ca.sin(theta6)*ca.cos(theta5) - 0.14*ca.sin(theta6)*ca.cos(theta3 + theta4) - 5.2491592393165e-34*ca.sin(theta6) - 0.14*ca.sin(theta3 + theta4)*ca.cos(theta5)*ca.cos(theta6) + 0.02198*ca.sin(theta3 + theta4) - 5.57214293612046e-18*ca.cos(theta5)*ca.cos(theta3 + theta4) + 5.57214293612046e-18*ca.cos(theta5) + 5.57214293612046e-18*ca.cos(theta3 + theta4) + 0.123

        o_x0 = 6.12323399573677e-17*((6.16297582203915e-33*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.sin(theta1 + theta2) + 6.12323399573676e-17*ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.sin(theta5) - ca.cos(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4))*ca.sin(theta6) - (6.16297582203915e-33*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.sin(theta1 + theta2) + 6.12323399573676e-17*ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.cos(theta5) - 6.12323399573677e-17*(6.12323399573677e-17*ca.sin(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.77373430684139e-49*ca.sin(theta1 + theta2)*ca.cos(theta5)*ca.cos(theta3 + theta4) + 6.12323399573677e-17*ca.sin(theta1 + theta2)*ca.cos(theta5) - 6.12323399573677e-17*ca.sin(theta1 + theta2) + 3.74939945665464e-33*ca.sin(theta3 + theta4)*ca.cos(theta5)*ca.cos(theta1 + theta2) + ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.cos(theta6) - 3.74939945665464e-33*ca.sin(theta1)*ca.cos(theta2) - 3.74939945665464e-33*ca.sin(theta2)*ca.cos(theta1) - ca.sin(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) - 3.06161699786838e-17*ca.sin(theta1 + theta2 - theta3 - theta4) + 3.06161699786838e-17*ca.sin(theta1 + theta2 + theta3 + theta4)

        o_y0 = -6.12323399573677e-17*((-6.12323399573676e-17*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) + 6.16297582203915e-33*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.cos(theta1 + theta2))*ca.sin(theta5) + ca.sin(theta1 + theta2)*ca.cos(theta5)*ca.cos(theta3 + theta4))*ca.sin(theta6) + (-6.12323399573676e-17*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) + 6.16297582203915e-33*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.cos(theta1 + theta2))*ca.cos(theta5) - 6.12323399573677e-17*(6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.74939945665464e-33*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4)*ca.cos(theta5) + ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) - 3.77373430684139e-49*ca.cos(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) - 6.12323399573677e-17*ca.cos(theta5)*ca.cos(theta1 + theta2) + 6.12323399573677e-17*ca.cos(theta1 + theta2))*ca.cos(theta6) - 3.74939945665464e-33*ca.sin(theta1)*ca.sin(theta2) - ca.sin(theta5)*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.74939945665464e-33*ca.cos(theta1)*ca.cos(theta2) + 3.06161699786838e-17*ca.cos(theta1 + theta2 - theta3 - theta4) - 3.06161699786838e-17*ca.cos(theta1 + theta2 + theta3 + theta4)

        o_z0 = 6.12323399573677e-17*(1 - ca.cos(theta3 + theta4))*ca.cos(theta5) + 6.12323399573677e-17*(-6.12323399573677e-17*(1 - ca.cos(theta3 + theta4))*ca.sin(theta5) + ca.sin(theta3 + theta4)*ca.cos(theta5))*ca.sin(theta6) + 6.12323399573677e-17*(6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta3 + theta4) - 3.74939945665464e-33*ca.cos(theta5)*ca.cos(theta3 + theta4) + 3.74939945665464e-33*ca.cos(theta5) - ca.cos(theta3 + theta4) - 3.74939945665464e-33)*ca.cos(theta6) - 6.12323399573677e-17*ca.sin(theta3)*ca.sin(theta4) + ca.sin(theta5)*ca.sin(theta3 + theta4) + 6.12323399573677e-17*ca.cos(theta3)*ca.cos(theta4) + 2.29584502165847e-49

        a_x0 = -((6.16297582203915e-33*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.sin(theta1 + theta2) + 6.12323399573676e-17*ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.sin(theta5) - ca.cos(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4))*ca.sin(theta6) - 6.12323399573677e-17*(6.16297582203915e-33*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.sin(theta1 + theta2) + 6.12323399573676e-17*ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.cos(theta5) + (6.12323399573677e-17*ca.sin(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.77373430684139e-49*ca.sin(theta1 + theta2)*ca.cos(theta5)*ca.cos(theta3 + theta4) + 6.12323399573677e-17*ca.sin(theta1 + theta2)*ca.cos(theta5) - 6.12323399573677e-17*ca.sin(theta1 + theta2) + 3.74939945665464e-33*ca.sin(theta3 + theta4)*ca.cos(theta5)*ca.cos(theta1 + theta2) + ca.sin(theta3 + theta4)*ca.cos(theta1 + theta2))*ca.cos(theta6) - 2.29584502165847e-49*ca.sin(theta1)*ca.cos(theta2) - 2.29584502165847e-49*ca.sin(theta2)*ca.cos(theta1) - 6.12323399573677e-17*ca.sin(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) - 1.87469972832732e-33*ca.sin(theta1 + theta2 - theta3 - theta4) + 1.87469972832732e-33*ca.sin(theta1 + theta2 + theta3 + theta4)

        a_y0 = ((-6.12323399573676e-17*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) + 6.16297582203915e-33*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.cos(theta1 + theta2))*ca.sin(theta5) + ca.sin(theta1 + theta2)*ca.cos(theta5)*ca.cos(theta3 + theta4))*ca.sin(theta6) + 6.12323399573677e-17*(-6.12323399573676e-17*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) + 6.16297582203915e-33*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) + ca.cos(theta1 + theta2))*ca.cos(theta5) + (6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + 3.74939945665464e-33*ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4)*ca.cos(theta5) + ca.sin(theta1 + theta2)*ca.sin(theta3 + theta4) - 3.77373430684139e-49*ca.cos(theta5)*ca.cos(theta1 + theta2)*ca.cos(theta3 + theta4) - 6.12323399573677e-17*ca.cos(theta5)*ca.cos(theta1 + theta2) + 6.12323399573677e-17*ca.cos(theta1 + theta2))*ca.cos(theta6) - 2.29584502165847e-49*ca.sin(theta1)*ca.sin(theta2) - 6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta1 + theta2)*ca.cos(theta3 + theta4) + 2.29584502165847e-49*ca.cos(theta1)*ca.cos(theta2) + 1.87469972832732e-33*ca.cos(theta1 + theta2 - theta3 - theta4) - 1.87469972832732e-33*ca.cos(theta1 + theta2 + theta3 + theta4)

        a_z0 = 3.74939945665464e-33*(1 - ca.cos(theta3 + theta4))*ca.cos(theta5) - ((-6.12323399573677e-17*(1 - ca.cos(theta3 + theta4))*ca.sin(theta5) + ca.sin(theta3 + theta4)*ca.cos(theta5))*ca.sin(theta6)) - ((6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta3 + theta4) - 3.74939945665464e-33*ca.cos(theta5)*ca.cos(theta3 + theta4) + 3.74939945665464e-33*ca.cos(theta5) - ca.cos(theta3 + theta4) - 3.74939945665464e-33)*ca.cos(theta6)) - 3.74939945665464e-33*ca.sin(theta3)*ca.sin(theta4) + 6.12323399573677e-17*ca.sin(theta5)*ca.sin(theta3 + theta4) + 3.74939945665464e-33*ca.cos(theta3)*ca.cos(theta4) + 1.40579962855621e-65
        
        fkine = ca.vertcat(x0,y0,z0,o_x0,o_y0,o_z0,a_x0,a_y0,a_z0)

        
        ## 定义系统状态和控制变量
        states = ca.vertcat(x,y,z,o_x,o_y,o_z,a_x,a_y,a_z,theta) # 实际上也可以通过 states = ca.vertcat(*[x, y, theta])一步实现
    
        n_states = states.size()[0] # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现 【这点很重要】
    
        ## 控制输入
        omega = ca.SX.sym('omega', 6, 1) # 6轴角速度
        controls = ca.vertcat(omega) # 控制向量
        n_controls = controls.size()[0] # 控制向量尺寸


        #leftpiper
        Jacobian = ca.jacobian(fkine, theta)
    

        # 运动学模型
        rhs = ca.vertcat(Jacobian@omega,omega)
        # 利用CasADi构建一个函数
        f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])


        ff,solver = self.init_solver(n_states,n_controls,N,f,T)
    


        # 开始仿真
        ## 定义约束条件，实际上CasADi需要在每次求解前更改约束条件。不过我们这里这些条件都是一成不变的
        ## 因此我们定义在while循环外，以提高效率
        ### 状态约束
        lbg = ca.DM(-np.pi*2)
        ubg = ca.DM(np.pi*2)
        ### 控制约束
        lbx = ca.DM(np.array([omega_max_1] * N).flatten())
        ubx = ca.DM(np.array([omega_max] * N).flatten())

        ## 仿真条件和相关变量
        t0 = 0.0 # 仿真时间
        t1 = 0.0
        u0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]*N).reshape(-1, 6) # 系统初始控制状态，为了统一本例中所有numpy有关
    

        x_c = [] # 存储系统的状态
        u_c = [] # 存储控制全部计算后的控制指令
        t_c = [] # 保存时间
        xx = [] # 存储每一步机器人位置
        index_t = [] # 存储时间戳，以便计算每一步求解的时间


        ## 开始仿真
        mpciter = 0 # 迭代计数器
        is_mpc = 0
        ### 终止条件为和目标的欧式距离小于0.01或者仿真超时
        # grp.open() # 夹爪打开
        print("start move")
        # print('x_right:', x_init, '\nxs_right:', xs)
        while(np.linalg.norm((x_init[:3])-(xs[:3]))>1e-1 or np.linalg.norm((x_init[3:9])-(xs[3:9]))>1e-0 ): 
        # while(np.linalg.norm((x_init[9:])-(xs[9:]))>1e-10): 
            ### 初始化优化参数
            is_mpc = 1
            c_p = np.concatenate((x_init, xs))
            
            ### 初始化优化目标变量
            init_control = ca.reshape(u0, -1, 1)
        
            ### 计算结果并且
            t_ = time.time()
            res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        
            index_t.append(time.time()- t_)
        
            # print('solve time:',time.time()- t_)
            ### 获得最优控制结果u
            u_sol = ca.reshape(res['x'], n_controls, N) # 记住将其恢复U的形状定义
            ### 
            ff_value = ff(u_sol, c_p) # 利用之前定义ff函数获得根据优化后的结果
                                    # 之后N+1步后的状态(n_states, N+1)


            ### 存储结果
            x_c.append(ff_value)
            u_c.append(u_sol[:, 0])
            t_c.append(t0)

        
            t0, x_init, u0 = self.shift_movement(T, t0, x_init, u_sol, f)


            ### 存储位置
            x_init_list = x_init.flatten().tolist()


            # 重新转换为 CasADi 的 DM 对象，并指定形状
            x_init = ca.DM(x_init_list)
            x_init = x_init.reshape((-1, 1))  # 这里需要将形状作为元组传递

            self.move_arms(x_init[9:15].full(),grab)
            xs_array = np.array(xs).reshape(1, -1)
        
            # print('x_right:', x_init, '\nxs_right:', xs_array)
        
            # print('u_right:', u_sol[:, 0])
            
            xx.append(x_init.full())
        
            mpciter = mpciter + 1


        return is_mpc

if __name__ == "__main__":
    mpc = MPC_piper()
    mpc.init()
    
    #left
    left_theta = [0,0,0,0,0,0]
    left_state = mpc.fkine_piper(left_theta)
    x_init = np.hstack([left_state, left_theta]).reshape(-1,1)

    left_theta1 = [0,0, 0, 0, 0, 1]
    left_state1 = mpc.fkine_piper(left_theta1)
    xs = np.hstack([left_state1, left_theta1]).reshape(-1,1)

 


    print("x_init shape:", x_init.shape)
    print("xs shape:", xs.shape)
    mpc.ur_move(x_init,xs,0)

