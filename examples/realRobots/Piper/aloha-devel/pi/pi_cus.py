import rospy
import threading
import queue
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
# 全局共享状态（用于将相对动作转为绝对位置）
# ========================
latest_puppet_state = {
    'left': np.zeros(7),
    'right': np.zeros(7)
}
puppet_state_lock = threading.Lock()

# 全局动作队列（线程安全）
action_queue = queue.Queue(maxsize=100)  # 限制缓冲，避免过时动作堆积
shutdown_flag = threading.Event()

task_instruction = "First, lay the pink towel flat, then fold it."  # 示例任务指令


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

        # 检查是否所有 deque 都非空
        if any(len(d) == 0 for d in deques):
            return False

        # 获取最新时间戳中的最小值作为同步时间
        stamps = [d[-1].header.stamp.to_sec() for d in deques]
        frame_time = min(stamps)

        # 检查所有最新消息是否 >= frame_time（应总是成立，但双重保险）
        for d in deques:
            if d[-1].header.stamp.to_sec() < frame_time:
                return False

        # 弹出旧于 frame_time 的数据，并取第一个 >= frame_time 的
        def pop_until(deq):
            while len(deq) > 1 and deq[0].header.stamp.to_sec() < frame_time:
                deq.popleft()
            return deq.popleft()

        img_left = self.bridge.imgmsg_to_cv2(pop_until(self.img_left_deque), 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(pop_until(self.img_right_deque), 'passthrough')
        img_d455 = self.bridge.imgmsg_to_cv2(pop_until(self.img_d455_deque), 'passthrough')

        master_arm_left = pop_until(self.master_arm_left_deque)
        master_arm_right = pop_until(self.master_arm_right_deque)
        puppet_arm_left = pop_until(self.puppet_arm_left_deque)
        puppet_arm_right = pop_until(self.puppet_arm_right_deque)

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

    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_d455_callback(self, msg):
        self.img_d455_deque.append(msg)

    def img_left_depth_callback(self, msg):
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        self.img_right_depth_deque.append(msg)

    def img_d455_depth_callback(self, msg):
        self.img_d455_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        self.puppet_arm_left_deque.append(msg)
        # 更新全局共享状态（线程安全）
        with puppet_state_lock:
            latest_puppet_state['left'] = np.array(msg.position[:7])

    def puppet_arm_right_callback(self, msg):
        self.puppet_arm_right_deque.append(msg)
        with puppet_state_lock:
            latest_puppet_state['right'] = np.array(msg.position[:7])

    def robot_base_callback(self, msg):
        self.robot_base_deque.append(msg)

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
    
    # Image topics
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--img_d455_topic', type=str, default='/camera/color/image_raw')
    
    # Depth topics
    parser.add_argument('--img_left_depth_topic', type=str, default='/camera_l/depth/image_raw')
    parser.add_argument('--img_right_depth_topic', type=str, default='/camera_r/depth/image_raw')
    parser.add_argument('--img_d455_depth_topic', type=str, default='/camera/depth/image_raw')
    
    # Arm topics
    parser.add_argument('--master_arm_left_topic', type=str, default='/puppet/joint_left')
    parser.add_argument('--master_arm_right_topic', type=str, default='/puppet/joint_right')
    parser.add_argument('--puppet_arm_left_topic', type=str, default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', type=str, default='/puppet/joint_right')
    
    # Base topic
    parser.add_argument('--robot_base_topic', type=str, default='/odom')
    parser.add_argument('--use_robot_base', action='store_true', default=False)
    parser.add_argument('--use_depth_image', action='store_true', default=False)
    parser.add_argument('--frame_rate', type=int, default=30)
    
    args = parser.parse_args()
    return args


# ========================
# 推理线程：获取观测 → 请求策略 → 入队动作（相对动作）
# ========================
def inference_worker(ros_op: 'RosOperator', client: websocket_client_policy.WebsocketClientPolicy):
    global action_queue, shutdown_flag
    print("[Inference Thread] Started.")

    while not shutdown_flag.is_set() and not rospy.is_shutdown():
        result = ros_op.get_frame()
        if not result:
            time.sleep(0.01)
            continue

        (img_left, img_right, img_d455,
         img_left_depth, img_right_depth, img_d455_depth,
         puppet_arm_left, puppet_arm_right,
         master_arm_left, master_arm_right,
         robot_base) = result

        qpos = np.concatenate((np.array(puppet_arm_left.position[:7]), np.array(puppet_arm_right.position[:7])))
        state = qpos

        # try:
        #     cam_high = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_d455, 224, 224))
        #     cam_right_wist = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_right, 224, 224))
        #     cam_left_wist = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_left, 224, 224))
        # except Exception as e:
        #     rospy.logwarn(f"Image processing failed: {e}")
        #     continue

        observation = {
            "images": {
                "cam_high": img_d455,
                "cam_right_wrist": img_right,
                "cam_left_wrist": img_left,
            },
            "state": state,
            "prompt": task_instruction,
        }

        # print(img_d455.shape)

        try:
            response = client.infer(observation)
            action_chunk = np.copy(response["actions"])
            rospy.loginfo(f"Received action chunk of shape {action_chunk.shape}")
        except Exception as e:
            rospy.logerr(f"Policy inference failed: {e}")
            continue

        for i in range(5):
            if shutdown_flag.is_set():
                break
            try:
                # print(action_chunk[i][6])
                # print(action_chunk[i][13])
                # if action_chunk[i][6] < 0.001:
                #     action_chunk[i][6] = 0
                # else:
                #     action_chunk[i][6] = 0.1
                # if action_chunk[i][13] < 0.001:
                #     action_chunk[i][13] = 0
                # else:
                #     action_chunk[i][13] = 0.1
                action_queue.put(action_chunk[i], block=False)
            except queue.Full:
                # rospy.logwarn("Action queue full, dropping oldest action.")
                try:
                    # print(action_chunk[i][6])
                    # print(action_chunk[i][13])
                    # action_queue.get_nowait()
                    # if action_chunk[i][6] < 0.001:
                    #     action_chunk[i][6] = 0
                    # else:
                    #     action_chunk[i][6] = 0.1
                    # if action_chunk[i][13] < 0.001:
                    #     action_chunk[i][13] = 0
                    # else:
                    #     action_chunk[i][13] = 0.1
                    action_queue.put(action_chunk[i], block=False)
                except:
                    pass
        
        time.sleep(0.01)

    print("[Inference Thread] Exiting.")


# ========================
# 执行线程：从队列取相对动作 → 转为绝对位置 → 发布
# ========================
def execution_worker():
    global action_queue, shutdown_flag, latest_puppet_state, puppet_state_lock
    print("[Execution Thread] Started.")

    pub_left = rospy.Publisher("/master/joint_left", JointState, queue_size=10)
    pub_right = rospy.Publisher("/master/joint_right", JointState, queue_size=10)

    rate = rospy.Rate(200)

    while not shutdown_flag.is_set() and not rospy.is_shutdown():
        try:
            action = action_queue.get(timeout=0.1)  # shape: (14,)
        except queue.Empty:
            rate.sleep()
            continue

        # 获取当前 puppet 实际关节位置（线程安全）
        with puppet_state_lock:
            current_left = latest_puppet_state['left'].copy()
            current_right = latest_puppet_state['right'].copy()
    

        # 模型输出是相对动作 (delta)
        delta_left = action[:7]
        delta_right = action[7:14]

        # 计算目标绝对位置
        target_left = delta_left
        target_right = delta_right

        header = Header()
        header.stamp = rospy.Time.now()

        js_left = JointState()
        js_left.header = header
        js_left.name = [f"joint_{i}" for i in range(7)]
        js_left.position = target_left.tolist()

        js_right = JointState()
        js_right.header = header
        js_right.name = [f"joint_{i}" for i in range(7)]
        js_right.position = target_right.tolist()

        # 发布控制命令
        pub_left.publish(js_left)
        pub_right.publish(js_right)

        # print(target_left)

        rate.sleep()

    print("[Execution Thread] Exiting.")


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    args = get_arguments()
    ros_op = RosOperator(args)

    # 初始化策略客户端
    client = websocket_client_policy.WebsocketClientPolicy(host="192.168.189.13", port=8000)

    # 启动两个线程
    inference_thread = threading.Thread(target=inference_worker, args=(ros_op, client))
    execution_thread = threading.Thread(target=execution_worker)

    inference_thread.start()
    execution_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        shutdown_flag.set()
        inference_thread.join(timeout=2)
        execution_thread.join(timeout=2)
        rospy.signal_shutdown("User requested shutdown.")