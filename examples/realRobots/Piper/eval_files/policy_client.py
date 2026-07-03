"""
Piper 双臂异步推理客户端(starVLA 版)。

推理后端从 GR00T 换成 starVLA:StarVLAPolicy 是 GR00T PolicyClient 的同名替身
(ping / get_action),把 obs→examples、(T,14) 动作重排都收在这里;执行端
(robot_server.py 的 Ruckig + ensemble)与异步流水线不变。参数见 config/client_config.yaml。

运行(在 starVLA 根, PYTHONPATH=根):
    python deployment/piper/policy_client.py [config/client_config.yaml]
"""

import os
import sys
import json
import time
import threading

import cv2
import numpy as np
import yaml
from robotmq import RMQClient, serialize, deserialize
from pynput import keyboard

esc_pressed = False


class StarVLAPolicy:
    """
    GR00T PolicyClient 的同名替身。用 fold(aloha-agilex)训练时, server 输出 14D 与执行端
    同布局 [L_arm6, L_grip, R_arm6, R_grip] 且已反归一化, 故 reorder 默认 identity。
    (若换布局不同的 ckpt, 在 config 改 reorder + 同步改下面 _obs_to_example 的 state 拼接。)
    """

    def __init__(self, host="127.0.0.1", port=10093, unnorm_key=None,
                 send_state=False, bgr_to_rgb=False,
                 use_ddim=True, num_ddim_steps=10,
                 cam_order=None, image_size=None, reorder=None,
                 instruction=None):
        if not (cam_order and reorder):
            raise ValueError("cam_order / reorder 必填(见 client_config.yaml 的 adapter 段)")

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self.unnorm_key = unnorm_key or meta.get("default_unnorm_key")
        self.available_unnorm_keys = meta.get("available_unnorm_keys", [])

        self.send_state = send_state
        self.bgr_to_rgb = bgr_to_rgb
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.cam_order = list(cam_order)
        self.image_size = tuple(image_size) if image_size else None   # null=不预压, 传原图给 processor smart_resize
        self.reorder = list(reorder)
        self.instruction = instruction

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sv = cfg.get("starvla", {}) or {}
        ad = cfg.get("adapter", {}) or {}
        return cls(
            host=sv.get("host", "127.0.0.1"), port=sv.get("port", 10093),
            unnorm_key=sv.get("unnorm_key"), send_state=sv.get("send_state", False),
            use_ddim=sv.get("use_ddim", True), num_ddim_steps=sv.get("num_ddim_steps", 10),
            cam_order=ad.get("cam_order"), image_size=ad.get("image_size"),
            reorder=ad.get("reorder"), bgr_to_rgb=ad.get("bgr_to_rgb", False),
            instruction=cfg.get("instruction"),
        )

    def ping(self):
        return self.action_chunk_size is not None

    def get_action(self, obs):
        vla_input = {
            "examples": [self._obs_to_example(obs)],
            "do_sample": False, "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps, "unnorm_key": self.unnorm_key,
            "normalize_state": self.send_state,   # server(PiperPolicyWrapper)端用训练 transform 归一化 state
        }
        resp = self.client.predict_action(vla_input)
        if isinstance(resp, dict) and resp.get("ok") is False:
            raise RuntimeError(f"starVLA server error: {resp.get('error')}")
        chunk = np.asarray(resp["data"]["actions"], dtype=np.float32)[0]   # (T,14) 已反归一化
        assert chunk.shape[-1] == len(self.reorder), f"动作维度 {chunk.shape[-1]} != {len(self.reorder)}"
        return chunk[:, self.reorder], {"chunk_size": int(chunk.shape[0])}  # -> deploy 顺序

    def _obs_to_example(self, obs):
        imgs = [self._prep(np.asarray(obs["video"][k])[0, 0]) for k in self.cam_order]
        instr = self.instruction or obs["language"]["annotation.human.task_description"][0][0]
        example = {"image": imgs, "lang": str(instr)}
        if self.send_state:
            s = obs["state"]  # -> fold(aloha-agilex)训练布局 [L_arm6, L_grip, R_arm6, R_grip]
            state = np.concatenate([
                np.asarray(s["left_arm"][0, 0], dtype=np.float32),
                np.asarray(s["left_gripper"][0, 0], dtype=np.float32),
                np.asarray(s["right_arm"][0, 0], dtype=np.float32),
                np.asarray(s["right_gripper"][0, 0], dtype=np.float32),
            ])
            example["state"] = state.reshape(1, -1)   # raw state;归一化在 server(PiperPolicyWrapper)端
        return example

    def _prep(self, im):
        im = np.asarray(im)
        if self.bgr_to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:   # 压到 image_size(应=[224,224]方形), 对齐训练 dataloader _pack_sample(datasets.py:1384 硬编码 resize((224,224)))
            im = cv2.resize(im, self.image_size, interpolation=cv2.INTER_AREA)
        return im.astype(np.uint8)


def on_press(key):
    global esc_pressed
    try:
        if key == keyboard.Key.esc:
            esc_pressed = True
            return False
    except Exception:
        pass


class RemoteRobotEnv:
    """Per-thread RMQClient (keeps per-thread-socket intent: obs_collector thread and main loop use separate sockets)."""

    def __init__(self, server_ip, port=8888):
        self.endpoint = f"tcp://{server_ip}:{port}"
        self._local = threading.local()

    def _get_client(self):
        if not hasattr(self._local, "client"):
            name = f"piper_{threading.current_thread().name}"
            self._local.client = RMQClient(name, self.endpoint)
        return self._local.client

    def reset(self):
        reply = deserialize(self._get_client().request_with_data(
            "reset", serialize({"type": "reset"}), timeout_s=30.0))
        return reply["obs"], reply["server_time"]

    def get_obs(self):
        client = self._get_client()
        data_list, _ = client.peek_data("obs", -1)   # non-destructive latest frame
        while not data_list:                          # startup: obs not published yet
            time.sleep(0.01)
            data_list, _ = client.peek_data("obs", -1)
        reply = deserialize(data_list[-1])
        return reply["obs"], reply["server_time"]

    def send_chunk(self, chunk_data, policy_step):
        self._get_client().put_data("chunk", serialize({"chunk_data": chunk_data, "policy_step": policy_step}))
        return {"status": "ok"}


def main(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "client_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    policy = StarVLAPolicy.from_yaml(config_path)
    rs = cfg.get("robot_server", {}) or {}
    env = RemoteRobotEnv(server_ip=rs.get("ip", "192.168.189.8"), port=rs.get("port", 8888))
    obs_freq = cfg.get("obs_freq", 15)
    obs_dt = 1.0 / obs_freq

    print("正在连接 starVLA policy server 与机器人环境 ...")
    if not policy.ping():
        raise RuntimeError("无法连接到 starVLA Policy Server")
    print(f"[starVLA] action_chunk_size={policy.action_chunk_size}, unnorm_key={policy.unnorm_key}")

    obs, obs_time = env.reset()

    # 异步流水线:obs 采集线程 + 主循环推理
    next_obs = None
    next_obs_time = 0.0
    obs_lock = threading.Lock()
    fresh_obs = threading.Event()

    def obs_collector():
        nonlocal next_obs, next_obs_time
        while not esc_pressed:
            t0 = time.time()
            o, t = env.get_obs()
            with obs_lock:
                next_obs = o
                next_obs_time = t
                fresh_obs.set()
            time.sleep(max(0, obs_dt - (time.time() - t0)))

    threading.Thread(target=obs_collector, daemon=True).start()
    print(f"[Async] 流水线启动 (观测 {obs_freq}Hz + starVLA 推理)")

    latency_log = []
    try:
        step = 0
        policy_step = 0
        while not esc_pressed:
            step += 1
            policy_step += 1
            obs_latency = time.time() - obs_time

            t0 = time.time()
            chunk, info = policy.get_action(obs)   # (T,14) 已反归一化, deploy 顺序
            infer_time = time.time() - t0

            t1 = time.time()
            env.send_chunk(chunk, policy_step)     # 发整块, server 端 ensemble 按帧插值
            comm_time = time.time() - t1
            e2e_latency = time.time() - obs_time

            latency_log.append({
                "step": step, "obs_time": obs_time,
                "obs_latency": round(obs_latency, 4), "infer_time": round(infer_time, 4),
                "comm_time": round(comm_time, 4), "e2e_latency": round(e2e_latency, 4),
                "chunk_size": int(info.get("chunk_size", chunk.shape[0])),
            })
            print(f"  [{step:3d}] 推理 {infer_time:.3f}s | 通信 {comm_time*1000:.1f}ms | "
                  f"E2E {e2e_latency:.3f}s | chunk={chunk.shape[0]}")

            if not fresh_obs.wait(timeout=1.0):
                print("[警告] 观测采集超时")
                if esc_pressed:
                    break
            with obs_lock:
                obs = next_obs
                obs_time = next_obs_time
                fresh_obs.clear()

    except KeyboardInterrupt:
        print("\n用户中断运行")
    finally:
        if latency_log:
            log_path = f"latency_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(latency_log, f, indent=2, ensure_ascii=False)
            it = [m["infer_time"] for m in latency_log]
            et = [m["e2e_latency"] for m in latency_log]
            ct = [m["comm_time"] for m in latency_log]
            print(f"\n延迟统计 ({len(latency_log)} 步): "
                  f"推理 mean={np.mean(it):.3f}s | 通信 mean={np.mean(ct)*1000:.1f}ms | "
                  f"E2E mean={np.mean(et):.3f}s (min={np.min(et):.3f} max={np.max(et):.3f})")
        print("系统正在关闭 ...")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
