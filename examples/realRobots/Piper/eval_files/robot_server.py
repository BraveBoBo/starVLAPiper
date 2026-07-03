import os
from RosOperator import RosOperator, get_arguments
import move_piper as mp
import numpy as np
import time
import cv2
from pynput import keyboard
from robotmq import RMQServer, serialize, deserialize
from robotmq.utils import clear_shared_memory
import threading
import queue
import yaml
from collections import deque
from dataclasses import dataclass, field
from ruckig import Ruckig, InputParameter, OutputParameter, Result
import warnings

from smooth import make_chunk_smoother, make_target_blender

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局初始化与安全急停
# ==========================================
stop_event = threading.Event()
reset_event = threading.Event()   # set by reset_and_wait, consumed by servo thread (clear chunks + home)

def on_press(key):
    try:
        if key == keyboard.Key.esc:
            print("\n[E-STOP] 按下 ESC，触发柔性急停...")
            stop_event.set()
            return False
    except:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

args = get_arguments()
robot = RosOperator(args)

# ==========================================
# 1. 配置
# ==========================================
@dataclass
class ServoConfig:
    dof: int
    control_freq: int           # 100Hz
    target_update_freq: int     # 20Hz (模型频率)
    max_velocity: float
    max_acceleration: float
    max_jerk: float

    ensemble_history_len: int = 4
    ensemble_exp_decay: float = 0.5

    chunk_smoother: dict = field(default_factory=lambda: {"method": "none"})
    target_blender: dict = field(default_factory=lambda: {"method": "ensemble"})
    joint_dims: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_freq

    @property
    def update_interval_steps(self) -> int:
        return self.control_freq // self.target_update_freq

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if 'target_update_freq' not in data: data['target_update_freq'] = 20
        if 'ensemble_history_len' not in data: data['ensemble_history_len'] = 4
        if 'ensemble_exp_decay' not in data: data['ensemble_exp_decay'] = 0.5
        data.pop('lookahead_time', None)
        data.pop('model_dt', None)
        return cls(**data)

# ==========================================
# 2. Step-Aligned ACT Ensemble + Ruckig
# ==========================================
class RuckigACTThread(threading.Thread):
    def __init__(self, chunk_queue: queue.Queue, obs_collector, config: ServoConfig):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.obs_collector = obs_collector
        self.config = config

        self.otg = Ruckig(config.dof, config.control_dt)
        self.inp = InputParameter(config.dof)
        self.out = OutputParameter(config.dof)

        self.inp.max_velocity = [config.max_velocity] * config.dof
        self.inp.max_acceleration = [config.max_acceleration] * config.dof
        self.inp.max_jerk = [config.max_jerk] * config.dof

        self.chunk_history = deque(maxlen=config.ensemble_history_len)
        self.global_servo_step = 0

        self.chunk_smoother = make_chunk_smoother(config)   # 阶段② 块后处理
        self.blender = make_target_blender(config)          # 阶段③ 块边界续接

    def run(self):
        print(f"[RuckigACT] 100Hz 连续伺服启动 | ACT Ensemble")

        init_obs, _ = self.obs_collector.get_latest()
        while init_obs is None and not stop_event.is_set():
            time.sleep(0.01)
            init_obs, _ = self.obs_collector.get_latest()
        if stop_event.is_set(): return

        init_q = np.concatenate([
            init_obs["state"]["left_arm"][0, 0], 
            init_obs["state"]["left_gripper"][0, 0],
            init_obs["state"]["right_arm"][0, 0], 
            init_obs["state"]["right_gripper"][0, 0]
        ])

        self.inp.current_position = init_q.astype(np.float64).tolist()
        # self.inp.current_velocity = [0.0] * self.config.dof
        # self.inp.current_acceleration = [0.0] * self.config.dof

        while not stop_event.is_set():
            loop_start = time.time()

            # ---- 0. Reset handshake: discard stale chunks, home through Ruckig ----
            if reset_event.is_set():
                while not self.chunk_queue.empty():
                    try: self.chunk_queue.get_nowait()
                    except queue.Empty: break
                self.chunk_history.clear()
                self.inp.target_position = [0.0] * self.config.dof   # home
                reset_event.clear()   # ack: history cleared, target=home

            # 计算当前包含小数的模型步 (例如: 2.2, 2.4, 2.6...)
            # 这使得 100Hz 的循环能对 20Hz 的 Chunk 进行精准降维切片
            current_fractional_step = self.global_servo_step / self.config.update_interval_steps

            # 取整用作默认的 birth_step
            current_model_step = int(current_fractional_step)

            # ---- 1. 拉取新 Chunk ----
            while not self.chunk_queue.empty():
                try:
                    msg = self.chunk_queue.get_nowait()
                    self.chunk_history.append({
                        "data": self.chunk_smoother.smooth(msg["chunk_data"]),   # 阶段② 块后处理
                        # [FIX 1]: 优先使用客户端传来的 policy_step 保证严格对齐
                        "birth_model_step": msg.get("policy_step", current_model_step)
                    })
                except queue.Empty:
                    break

            # ---- 2. 100Hz 目标融合更新 (阶段③ 可切换 blender) ----
            #        (旧块由 chunk_history 的 maxlen deque 自动淘汰, 无需手动清理)
            if len(self.chunk_history) > 0:
                q_target = self.blender.target(self.chunk_history, current_fractional_step)
                if q_target is not None:
                    self.inp.target_position = q_target.tolist()
                    # self.inp.target_velocity = [0.0] * self.config.dof
                    # self.inp.target_acceleration = [0.0] * self.config.dof

            # ---- 3. Ruckig 限幅 ----
            result = self.otg.update(self.inp, self.out)
            if result in (Result.Working, Result.Finished):
                cmd = np.array(self.out.new_position, dtype=np.float32)
                mp.move_arms(cmd[0:7], cmd[7:14])
                self.out.pass_to_input(self.inp)

            # ---- 4. 步数推进与定时 ----
            self.global_servo_step += 1
            elapsed = time.time() - loop_start
            time.sleep(max(0, self.config.control_dt - elapsed))

        # ---- 柔性急停 ----
        print("[RuckigACT] 执行急停制动...")
        self.inp.target_velocity = [0.0] * self.config.dof
        self.inp.target_acceleration = [0.0] * self.config.dof
        self.inp.target_position = self.inp.current_position 
        t_stop = time.time()
        while time.time() - t_stop < 0.5:
            self.otg.update(self.inp, self.out)
            cmd = np.array(self.out.new_position, dtype=np.float32)
            mp.move_arms(cmd[0:7], cmd[7:14])
            self.out.pass_to_input(self.inp)
            time.sleep(self.config.control_dt)

# ==========================================
# 3. 观测采集线程 (免阻碍读取)
# ==========================================
class ObsCollectorThread(threading.Thread):
    def __init__(self, language_instruction="fold the white T-shirt"):
        super().__init__(daemon=True)
        self.language_instruction = language_instruction
        self._lock = threading.Lock()
        self._cached_obs = None
        self._cached_time = 0.0

    def get_latest(self):
        with self._lock:
            return self._cached_obs, self._cached_time

    def _capture_once(self):
        result = robot.get_frame()
        if result is False: return None

        capture_time = time.time()
        img_left = result[0][None, None, ..., :]
        img_right = result[1][None, None, ..., :]
        img_d455 = cv2.resize(result[2], (640, 480))[None, None, ..., :]

        left_arm = np.array(result[6].position[0:6], dtype=np.float32)[None, None, ...]
        left_gripper = np.array(result[6].position[6:7], dtype=np.float32)[None, None, ...]
        right_arm = np.array(result[7].position[0:6], dtype=np.float32)[None, None, ...]
        right_gripper = np.array(result[7].position[6:7], dtype=np.float32)[None, None, ...]

        obs = {
            "video": {
                "cam_d455": img_d455,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "state": {
                "left_arm": left_arm,
                "left_gripper": left_gripper,
                "right_arm": right_arm,
                "right_gripper": right_gripper,
            },
            "language": {
                "annotation.human.task_description": [[self.language_instruction]],
            }
        }
        return obs, capture_time

    def run(self):
        while not stop_event.is_set():
            result = self._capture_once()
            if result is not None:
                obs, t = result
                with self._lock:
                    self._cached_obs = obs
                    self._cached_time = t

    def reset_and_wait(self):
        # signal the servo thread to home (it discards stale chunks and drives target=home through
        # Ruckig); keeps move_arms single-writer and get_frame single-consumer (obs thread only)
        reset_event.set()
        while reset_event.is_set() and not stop_event.is_set():
            time.sleep(0.01)

        # wait until the arm actually reaches home (obs state), 10s timeout fallback
        deadline = time.time() + 10.0
        while not stop_event.is_set():
            obs, _ = self.get_latest()
            if obs is not None:
                q = np.concatenate([
                    obs["state"]["left_arm"][0, 0], obs["state"]["left_gripper"][0, 0],
                    obs["state"]["right_arm"][0, 0], obs["state"]["right_gripper"][0, 0],
                ])
                if np.abs(q).max() < 0.05:
                    break
            if time.time() > deadline:
                print("[Reset] homing timeout (10s), returning current pose")
                break
            time.sleep(0.05)

        # return a frame captured after homing settled
        t_home = time.time()
        while not stop_event.is_set():
            obs, t = self.get_latest()
            if obs is not None and t >= t_home:
                return obs, t
            time.sleep(0.02)

# ==========================================
# 4. robotmq 通信主干
# ==========================================
def main():
    clear_shared_memory()  # clear crashed-server leftovers in /dev/shm
    server = RMQServer("piper_robot", "tcp://0.0.0.0:8888")
    server.add_topic("obs", 1.0)     # robot->client latest obs (client peek n=-1); TTL 1s drops stale frames
    server.add_topic("chunk", 1.0)   # client->robot action chunks (drained pop n=0)
    server.add_topic("reset", 5.0)   # sync reset RPC

    cfg = ServoConfig.from_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "robot_config.yaml"))

    # [FIX 2]: tiny queue, no backlog, force exec thread to keep cadence
    chunk_queue = queue.Queue(maxsize=3)

    # init arm controllers in the main thread: the first puppet-feedback wait (or its
    # RuntimeError) must not land inside a 100Hz servo tick (see move_piper.init_arms)
    mp.init_arms()

    obs_collector = ObsCollectorThread()
    obs_collector.start()

    exec_thread = RuckigACTThread(chunk_queue, obs_collector, cfg)
    exec_thread.start()

    print("[RobotServer] Ready on port 8888 (robotmq, Fractional Step-Aligned Ensemble Active)")

    last_obs_time = -1.0
    while not stop_event.is_set():
        try:
            # publish latest obs on each new captured frame (client peek n=-1 = latest, replaces old get_obs REQ)
            obs, capture_time = obs_collector.get_latest()
            if obs is not None and capture_time != last_obs_time:
                server.put_data("obs", serialize({"obs": obs, "server_time": capture_time}))
                last_obs_time = capture_time

            # drain chunk topic -> chunk_queue (drop-oldest, same as old send_chunk)
            chunks, _ = server.pop_data("chunk", 0)
            for cb in chunks:
                msg = deserialize(cb)
                if chunk_queue.full():
                    try: chunk_queue.get_nowait()
                    except queue.Empty: pass
                chunk_queue.put_nowait({
                    "chunk_data": msg["chunk_data"],
                    "policy_step": msg.get("policy_step", None),
                })

            # reset RPC; 5ms timeout also paces this loop (~200Hz)
            _, topic = server.wait_for_request(0.005)
            if topic == "reset":
                obs, capture_time = obs_collector.reset_and_wait()
                server.reply_request("reset", serialize({"obs": obs, "server_time": capture_time}))
        except Exception as e:
            if not stop_event.is_set():
                print(f"[robotmq Error] {e}")

    obs_collector.join(timeout=1.0)
    exec_thread.join(timeout=1.0)
    print("[RobotServer] 优雅退出。")

if __name__ == "__main__":
    main()
