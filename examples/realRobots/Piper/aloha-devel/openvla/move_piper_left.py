#! /usr/bin/env python
import rospy
import numpy as np
import actionlib
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

def move_arms(left_joint_positions,grab):
    # 初始化节点，名称为'talker'，anonymous=True确保节点名称唯一
    rospy.init_node('talker', anonymous=True)
    
    # 创建publisher，发布到'chatter'话题，消息类型为String，队列大小10
    pub = rospy.Publisher('/master/joint_left', JointState, queue_size=10)

    pos = np.append(left_joint_positions, grab)
    # 设置发布频率为10Hz
    rate = rospy.Rate(10) # 10hz
    
    # 当节点没有被关闭时
    while not rospy.is_shutdown():
        # 要发布的消息内容
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = pos
        
        # 打印日志信息
        rospy.loginfo(joint_state_msg)
        
        # 发布消息
        pub.publish(joint_state_msg)

        # 按照设定的频率休眠
        rate.sleep()

if __name__ == '__main__':
    try:
        left_joint_positions=[0, 0, 0, 0, 0, 0,0.0]

        move_arms(left_joint_positions)
    except rospy.ROSInterruptException:
        # 捕获ROS中断异常，正常退出
        pass