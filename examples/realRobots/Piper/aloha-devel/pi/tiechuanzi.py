import rospy
import time
import numpy as np
from collections import deque
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import cv2
import argparse

# 假设你已有这些模块
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# ========================
# 全局：当前 puppet 状态（用于 delta → absolute）
# ========================
latest_puppet_state = {
    'left': np.zeros(7),
    'right': np.zeros(7)
}

task_instruction = "First, lay the pink towel flat, then fold it."

class RosOperator:
    def __init__(self, args):
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque(maxlen=2000)
        self.img_right_deque = deque(maxlen=2000)
        self.img_d455_deque = deque(maxlen=2000)
        self.img_left_depth_deque = deque(maxlen=2000)
        self.img_right_depth_deque = deque(maxlen=2000)
        self.img_d455_depth_deque = deque(maxlen=2000)

        self.master_arm_left_deque = deque(maxlen=2000)
        self.master_arm_right_deque = deque(maxlen=2000)
        self.puppet_arm_left_deque = deque(maxlen=2000)
        self.puppet_arm_right_deque = deque(maxlen=2000)
        self.robot_base_deque = deque(maxlen=2000)

    def get_frame(self):
        deques = [
            self.img_left_deque,
            self.img_right_deque,
            self.img_d455_deque,
            self.master_arm_left_deque,
            self.master_arm_right_deque,
            self.puppet_arm_left_deque,
            self.puppet_arm_right_deque,
        ]
        if self.args.use_depth_image:
            deques.extend([
                self.img_left_depth_deque,
                self.img_right_depth_deque,
                self.img_d455_depth_deque,
            ])
        if self.args.use_robot_base:
            deques.append(self.robot_base_deque)

        if any(len(d) == 0 for d in deques):
            return None

        stamps = [d[-1].header.stamp.to_sec() for d in deques]
        frame_time = min(stamps)

        for d in deques:
            if d[-1].header.stamp.to_sec() < frame_time:
                return None

        def pop_until(deq):
            while len(deq) > 1 and deq[0].header.stamp.to_sec() < frame_time:
                deq.popleft()
            return deq.popleft()

        try:
            img_left = self.bridge.imgmsg_to_cv2(pop_until(self.img_left_deque), 'passthrough')
            img_right = self.bridge.imgmsg_to_cv2(pop_until(self.img_right_deque), 'passthrough')
            img_d455 = self.bridge.imgmsg_to_cv2(pop_until(self.img_d455_deque), 'passthrough')

            master_arm_left = pop_until(self.master_arm_left_deque)
            master_arm_right = pop_until(self.master_arm_right_deque)
            puppet_arm_left = pop_until(self.puppet_arm_left_deque)
            puppet_arm_right = pop_until(self.puppet_arm_right_deque)

            # 更新全局 puppet 状态（关键！）
            latest_puppet_state['left'] = np.array(puppet_arm_left.position[:7])
            latest_puppet_state['right'] = np.array(puppet_arm_right.position[:7])

            img_left_depth = img_right_depth = img_d455_depth = None
            if self.args.use_depth_image:
                img_left_depth = self.bridge.imgmsg_to_cv2(pop_until(self.img_left_depth_deque), 'passthrough')
                img_left_depth = cv2.copyMakeBorder(img_left_depth, 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=0)
                img_right_depth = self.bridge.imgmsg_to_cv2(pop_until(self.img_right_depth_deque), 'passthrough')
                img_right_depth = cv2.copyMakeBorder(img_right_depth, 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=0)
                img_d455_depth = self.bridge.imgmsg_to_cv2(pop_until(self.img_d455_depth_deque), 'passthrough')

            robot_base = None
            if self.args.use_robot_base:
                robot_base = pop_until(self.robot_base_deque)

            return (
                img_left, img_right, img_d455,
                img_left_depth, img_right_depth, img_d455_depth,
                puppet_arm_left, puppet_arm_right,
                master_arm_left, master_arm_right,
                robot_base
            )
        except Exception as e:
            rospy.logwarn(f"Failed to get frame: {e}")
            return None

    # === Callbacks ===
    def img_left_callback(self, msg): self.img_left_deque.append(msg)
    def img_right_callback(self, msg): self.img_right_deque.append(msg)
    def img_d455_callback(self, msg): self.img_d455_deque.append(msg)
    def img_left_depth_callback(self, msg): self.img_left_depth_deque.append(msg)
    def img_right_depth_callback(self, msg): self.img_right_depth_deque.append(msg)
    def img_d455_depth_callback(self, msg): self.img_d455_depth_deque.append(msg)
    def master_arm_left_callback(self, msg): self.master_arm_left_deque.append(msg)
    def master_arm_right_callback(self, msg): self.master_arm_right_deque.append(msg)
    def puppet_arm_left_callback(self, msg): self.puppet_arm_left_deque.append(msg)
    def puppet_arm_right_callback(self, msg): self.puppet_arm_right_deque.append(msg)
    def robot_base_callback(self, msg): self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_d455_topic, Image, self.img_d455_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_d455_depth_topic, Image, self.img_d455_depth_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="./data")
    parser.add_argument('--task_name', type=str, default="aloha_mobile_dummy")
    parser.add_argument('--episode_idx', type=int, default=0)
    parser.add_argument('--max_timesteps', type=int, default=500)
    parser.add_argument('--camera_names', nargs='+', default=['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_d455'])
    
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--img_d455_topic', type=str, default='/camera/color/image_raw')
    
    parser.add_argument('--img_left_depth_topic', type=str, default='/camera_l/depth/image_raw')
    parser.add_argument('--img_right_depth_topic', type=str, default='/camera_r/depth/image_raw')
    parser.add_argument('--img_d455_depth_topic', type=str, default='/camera/depth/image_raw')
    
    parser.add_argument('--master_arm_left_topic', type=str, default='/puppet/joint_left')
    parser.add_argument('--master_arm_right_topic', type=str, default='/puppet/joint_right')
    parser.add_argument('--puppet_arm_left_topic', type=str, default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', type=str, default='/puppet/joint_right')
    
    parser.add_argument('--robot_base_topic', type=str, default='/odom')
    parser.add_argument('--use_robot_base', action='store_true', default=False)
    parser.add_argument('--use_depth_image', action='store_true', default=False)
    parser.add_argument('--frame_rate', type=int, default=30)
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_op = RosOperator(args)

    # 初始化策略客户端
    client = websocket_client_policy.WebsocketClientPolicy(host="192.168.189.13", port=8000)

    # 创建发布者
    pub_left = rospy.Publisher("/master/joint_left", JointState, queue_size=10)
    pub_right = rospy.Publisher("/master/joint_right", JointState, queue_size=10)
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    header = Header()
    header.stamp = rospy.Time.now()
    # print(left0)
    js_left = JointState(header=header, name=[f"joint_{i}" for i in range(7)], position=left0)
    js_right = JointState(header=header, name=[f"joint_{i}" for i in range(7)], position=right0)
    pub_left.publish(js_left)
    pub_right.publish(js_right)


    rospy.loginfo("Starting serial control loop...")

    try:
        while not rospy.is_shutdown():
            # Step 1: 获取同步观测帧
            result = ros_op.get_frame()
            if result is None:
                continue

            (img_left, img_right, img_d455,
             img_left_depth, img_right_depth, img_d455_depth,
             puppet_arm_left, puppet_arm_right,
             master_arm_left, master_arm_right,
             robot_base) = result

            # Step 2: 构造 observation
            qpos = np.concatenate((
                np.array(puppet_arm_left.position[:7]),
                np.array(puppet_arm_right.position[:7])
            ))

            observation = {
                "images": {
                    "cam_high": img_d455,
                    "cam_right_wrist": img_right,
                    "cam_left_wrist": img_left,
                },
                "state": qpos,
                "prompt": task_instruction,
            }

            try:
                response = client.infer(observation)
                action = np.copy(response["actions"]) 
                # for i in range(action.shape[0]):
                #     if action[i][6] < 0.0095:
                #         action[i][6] = 0
                #     else:
                #         action[i][6] = 0.1
                #     if action[i][13] < 0.0095:
                #         action[i][13] = 0
                #     else:
                #         action[i][13] = 0.1
            except Exception as e:
                rospy.logerr(f"Policy inference failed: {e}")
                continue


            N=25
            M=30

            # Step 1: 取前 N 步
            if action.shape[0] < N:
                rospy.logwarn(f"Action length ({action.shape[0]}) < N ({N}), skipping.")
                continue
            action_subset = action[:N]  # shape: (N, 14)

            # Step 2: 插值到 M 步（在时间维度上均匀重采样）
            # 原始时间索引: 0, 1, 2, ..., N-1
            # 目标时间索引: 0, (N-1)/(M-1), 2*(N-1)/(M-1), ..., N-1
            original_t = np.linspace(0, 1, N)
            target_t = np.linspace(0, 1, M)

            # 对每个关节维度单独插值
            interp_action = np.zeros((M, 14))
            for j in range(14):
                interp_action[:, j] = np.interp(target_t, original_t, action_subset[:, j])

            # Step 3: 以 200 Hz 执行插值后的 M 步
            rate = rospy.Rate(200)
            for i in range(M):
                if rospy.is_shutdown():
                    break

                act = interp_action[i]
                target_left = act[:7]
                target_right = act[7:14]

                header = Header()
                header.stamp = rospy.Time.now()

                js_left = JointState(
                    header=header,
                    name=[f"joint_{idx}" for idx in range(7)],
                    position=target_left.tolist()
                )
                js_right = JointState(
                    header=header,
                    name=[f"joint_{idx}" for idx in range(7)],
                    position=target_right.tolist()
                )

                pub_left.publish(js_left)
                pub_right.publish(js_right)
                rate.sleep()
                
            # rate = rospy.Rate(200)

            # for i in range(5):


            #     target_left = action[i][:7]   
            #     target_right = action[i][7:14]

            #     # Step 5: 发布控制命令
            #     header = Header()
            #     header.stamp = rospy.Time.now()

            #     js_left = JointState(header=header, name=[f"joint_{i}" for i in range(7)], position=target_left.tolist())
            #     js_right = JointState(header=header, name=[f"joint_{i}" for i in range(7)], position=target_right.tolist())

            #     pub_left.publish(js_left)
            #     pub_right.publish(js_right)
            #     rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        rospy.signal_shutdown("Serial control loop ended.")


if __name__ == "__main__":
    main()