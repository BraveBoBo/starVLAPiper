#! /usr/bin/env python
import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

# 关节名称
LEFT_JOINT_NAMES = ['joint0', 'joint1', 'joint2',
                    'joint3', 'joint4', 'joint5']
RIGHT_JOINT_NAMES = ['joint0', 'fr_joint2', 'fr_joint3',
                    'fr_joint4', 'fr_joint5', 'fr_joint6']

# 机械臂控制客户端
left_arm_client = None
right_arm_client = None

def init_clients():
    global left_arm_client, right_arm_client
    
    rospy.init_node("dual_arm_controller", anonymous=True)
    if left_arm_client is None:
        left_arm_client = actionlib.SimpleActionClient(
            '/master/joint_left', FollowJointTrajectoryAction)
        print("Waiting for left arm server...")
        # left_arm_client.wait_for_server()
        print("Left arm connected.")
    if right_arm_client is None:
        right_arm_client = actionlib.SimpleActionClient(
            '/master/joint_right',FollowJointTrajectoryAction)  
        print("Waiting for right arm server...")
        # right_arm_client.wait_for_server()
        print("Right arm connected.")

def move_arms(left_joint_positions, right_joint_positions):
    """
    同时发送目标关节位置给左右机械臂
    :param left_joint_positions: 左臂 6 关节目标角度
    :param right_joint_positions: 右臂 6 关节目标角度
    """
    init_clients()
    left_goal = FollowJointTrajectoryGoal()
    left_goal.trajectory = JointTrajectory()
    left_goal.trajectory.joint_names = LEFT_JOINT_NAMES

    right_goal = FollowJointTrajectoryGoal()
    right_goal.trajectory = JointTrajectory()
    right_goal.trajectory.joint_names = RIGHT_JOINT_NAMES

    left_trajectory_points_array = (left_joint_positions[0],left_joint_positions[1],left_joint_positions[2],left_joint_positions[3],left_joint_positions[4],left_joint_positions[5])
    left_goal.trajectory.points = [JointTrajectoryPoint(positions=left_trajectory_points_array, velocities=[0]*6, time_from_start=rospy.Duration(0.02))]

    right_trajectory_points_array = (right_joint_positions[0],right_joint_positions[1],right_joint_positions[2],right_joint_positions[3],right_joint_positions[4],right_joint_positions[5])
    right_goal.trajectory.points = [JointTrajectoryPoint(positions=right_trajectory_points_array, velocities=[0]*6, time_from_start=rospy.Duration(0.02))]



    # **同时发送目标**
    left_arm_client.send_goal(left_goal)
    right_arm_client.send_goal(right_goal)

    # **同时等待结果**
    '''left_arm_client.wait_for_result()
    right_arm_client.wait_for_result()
    
    if left_arm_client.get_result():
        print("Left arm motion completed successfully.")
    else:
        print("Left arm motion failed.")
    
    if right_arm_client.get_result():
        print("Right arm motion completed successfully.")
    else:
        print("Right arm motion failed.")'''

if __name__ == "__main__":
    init_clients()
    home = 0
    # **同时控制两个机械臂**
    if home:
        move_arms(
            left_joint_positions=[0, 0, 0, 0, 0, 0],
            right_joint_positions=[0, 0, 0, 0, 0, 0]
        )
    else:
        move_arms(
            left_joint_positions=[-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875],
            right_joint_positions=[-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        )
