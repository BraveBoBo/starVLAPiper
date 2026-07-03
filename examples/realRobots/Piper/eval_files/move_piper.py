#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import threading
import time

class AsyncArmController:
    def __init__(self, arm_side):  # 'left' or 'right'
        self.arm_side = arm_side
        self.feedback_topic = f"/puppet/joint_{arm_side}"
        self.control_topic = f"/master/joint_{arm_side}"

        self.lock = threading.Lock()
        self.latest_target = None
        self.current_pos = None
        self.running = True

        self.pub = rospy.Publisher(self.control_topic, JointState, queue_size=10)
        self.sub = rospy.Subscriber(self.feedback_topic, JointState, self._feedback_callback)

        self._wait_for_initial_pos()

        self.pub_thread = threading.Thread(target=self._background_publish, daemon=True)
        self.pub_thread.start()
        rospy.loginfo(f"AsyncArmController ({arm_side} arm) 初始化完成，等待目标指令...")

    def _feedback_callback(self, msg):
        joint_order = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        with self.lock:
            try:
                self.current_pos = [msg.position[msg.name.index(j)] for j in joint_order]
            except ValueError as e:
                rospy.logwarn(f"[{self.arm_side}] 关节名不匹配: {e}")

    def _wait_for_initial_pos(self):
        wait_cnt = 0
        while self.current_pos is None and not rospy.is_shutdown():
            if wait_cnt % 5 == 0:
                rospy.loginfo(f"等待 puppet 的 {self.arm_side} 臂状态（话题：{self.feedback_topic}）...")
            time.sleep(0.2)
            wait_cnt += 1
            if wait_cnt > 30:
                raise RuntimeError(f"超时：未从 {self.feedback_topic} 获取到 {self.arm_side} 臂状态")

    def _background_publish(self):
        rate = rospy.Rate(100)   # 对齐 robot_server 的 100Hz 伺服，避免丢中间 setpoint
        while self.running and not rospy.is_shutdown():
            with self.lock:
                if self.latest_target is not None:
                    cmd_msg = JointState()
                    cmd_msg.header.stamp = rospy.Time.now()
                    cmd_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                    cmd_msg.position = self.latest_target
                    self.pub.publish(cmd_msg)
            rate.sleep()

    def update_target(self, target_7d):
        if len(target_7d) != 7:
            rospy.logerr(f"[{self.arm_side}] 目标需7个关节值，当前{len(target_7d)}个，跳过！")
            return False
        with self.lock:
            self.latest_target = [float(x) for x in target_7d]  # numpy(float32) -> python float list, 供 JointState.position 序列化
        # rospy.logdebug(f"[{self.arm_side}] 更新目标: {[(p,4) for p in self.latest_target]}")
        return True

    def stop(self):
        with self.lock:
            self.running = False


# 全局双臂控制器单例
_arm_controllers = {'left': None, 'right': None}

def _ensure_controllers():
    """确保 ROS 节点已初始化、左右臂 controller 已创建（创建时会阻塞等一次 puppet 反馈）。"""
    global _arm_controllers
    if not rospy.core.is_initialized():
        rospy.init_node("async_dual_arm_node", anonymous=True)
    if _arm_controllers['left'] is None:
        _arm_controllers['left'] = AsyncArmController('left')
    if _arm_controllers['right'] is None:
        _arm_controllers['right'] = AsyncArmController('right')


def init_arms():
    """预初始化双臂 controller。在伺服循环启动前调用，让首次等待 puppet 反馈的阻塞/超时
    发生在启动阶段，而不是 100Hz 伺服线程的某个 tick 里（否则 RuntimeError 会打挂伺服线程）。"""
    _ensure_controllers()


def move_arms(left_target_7d, right_target_7d):
    """
    MPC 调用接口：同时控制左右臂（各 7 个关节值，含夹爪）。非阻塞：只更新目标。
    参数:
        left_target_7d: list of 7 floats
        right_target_7d: list of 7 floats
    """
    _ensure_controllers()
    _arm_controllers['left'].update_target(left_target_7d)
    _arm_controllers['right'].update_target(right_target_7d)


# 示例：主程序测试双臂归零
if __name__ == "__main__":
    try:
        # 归零：左右臂都设为 [0,0,0,0,0,0,0]
        left = [-0.07998818159103394, 1.7889211177825928, -1.6046841144561768, -0.7669841051101685, 0.858552098274231, 0.9722305536270142, 0.020082473754882812]
        right = [-0.02821110188961029, 0.052703022956848145, -0.041123032569885254, 0.3149459958076477, -0.005659997463226318, 0.1034461110830307, 0.0006692837923765182]
        move_arms([0]*7, [0,0,0,0,0,0,0])
        # move_arms(left, right)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass