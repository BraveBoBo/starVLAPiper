import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import deque
import base64
from PIL import Image
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

class ActionVisualizer:
    def __init__(self, max_history=100):
        """
        动作可视化器
        max_history: 保存的历史动作数量
        """
        self.max_history = max_history
        self.action_history = deque(maxlen=max_history)
        self.time_stamps = deque(maxlen=max_history)
        self.current_step = 0
        
    def add_action(self, action, timestamp=None):
        """
        添加新的动作数据
        action: 动作数组 [x, y, z, roll, pitch, yaw, gripper]
        """
        if timestamp is None:
            timestamp = self.current_step
        
        self.action_history.append(action)
        self.time_stamps.append(timestamp)
        self.current_step += 1
        
    def plot_action_trajectory(self, save_path=None):
        """
        绘制动作轨迹
        """
        if len(self.action_history) == 0:
            print("没有动作数据可供可视化")
            return
            
        actions = np.array(list(self.action_history))
        times = np.array(list(self.time_stamps))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('机器人动作轨迹可视化', fontsize=16)
        
        # 位置轨迹 (x, y, z)
        position_labels = ['X位置 (m)', 'Y位置 (m)', 'Z位置 (m)']
        for i in range(3):
            axes[0, i].plot(times, actions[:, i], 'b-', linewidth=2)
            axes[0, i].set_title(position_labels[i])
            axes[0, i].set_xlabel('时间步')
            axes[0, i].grid(True)
            
        # 姿态轨迹 (roll, pitch, yaw)
        orientation_labels = ['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        for i in range(3):
            axes[0, 3].plot(times, actions[:, i+3], label=orientation_labels[i], linewidth=2)
        axes[0, 3].set_title('姿态角度')
        axes[0, 3].set_xlabel('时间步')
        axes[0, 3].legend()
        axes[0, 3].grid(True)
        
        # 夹爪状态
        if actions.shape[1] > 6:
            axes[1, 0].plot(times, actions[:, 6], 'r-', linewidth=2)
            axes[1, 0].set_title('夹爪状态')
            axes[1, 0].set_xlabel('时间步')
            axes[1, 0].set_ylabel('夹爪开合度')
            axes[1, 0].grid(True)
        
        # 3D轨迹
        ax_3d = fig.add_subplot(2, 4, 6, projection='3d')
        ax_3d.plot(actions[:, 0], actions[:, 1], actions[:, 2], 'b-', linewidth=2)
        ax_3d.scatter(actions[0, 0], actions[0, 1], actions[0, 2], c='g', s=100, label='起点')
        ax_3d.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2], c='r', s=100, label='终点')
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D轨迹')
        ax_3d.legend()
        
        # 速度分析
        if len(actions) > 1:
            velocities = np.diff(actions[:, :3], axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            axes[1, 2].plot(times[1:], speeds, 'g-', linewidth=2)
            axes[1, 2].set_title('运动速度')
            axes[1, 2].set_xlabel('时间步')
            axes[1, 2].set_ylabel('速度 (m/step)')
            axes[1, 2].grid(True)
        
        # 动作变化率
        if len(actions) > 1:
            action_changes = np.diff(actions, axis=0)
            action_change_norm = np.linalg.norm(action_changes, axis=1)
            axes[1, 3].plot(times[1:], action_change_norm, 'purple', linewidth=2)
            axes[1, 3].set_title('动作变化率')
            axes[1, 3].set_xlabel('时间步')
            axes[1, 3].set_ylabel('变化幅度')
            axes[1, 3].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        
    def plot_real_time(self, action, image=None):
        """
        实时可视化当前动作
        """
        self.add_action(action)
        
        # 创建实时显示窗口
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示当前图像
        if image is not None:
            if isinstance(image, str):  # base64编码的图像
                image_data = base64.b64decode(image)
                image = Image.open(io.BytesIO(image_data))
                image = np.array(image)
            
            axes[0].imshow(image)
            axes[0].set_title('当前图像')
            axes[0].axis('off')
        
        # 显示当前动作值
        action_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
        y_pos = np.arange(len(action))
        
        axes[1].barh(y_pos, action, color=['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink'])
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(action_labels)
        axes[1].set_xlabel('数值')
        axes[1].set_title('当前动作值')
        axes[1].grid(True, axis='x')
        
        # 显示历史轨迹
        if len(self.action_history) > 1:
            actions = np.array(list(self.action_history))
            times = np.array(list(self.time_stamps))
            
            # 只显示位置轨迹
            axes[2].plot(times, actions[:, 0], 'r-', label='X', linewidth=2)
            axes[2].plot(times, actions[:, 1], 'g-', label='Y', linewidth=2)
            axes[2].plot(times, actions[:, 2], 'b-', label='Z', linewidth=2)
            axes[2].set_xlabel('时间步')
            axes[2].set_ylabel('位置 (m)')
            axes[2].set_title('位置历史轨迹')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新显示
        

class MultiCameraVisualizer:
    def __init__(self, camera_names):
        """
        多相机图像可视化器
        camera_names: 相机名称列表
        """
        self.camera_names = camera_names
        
    def display_multi_camera(self, image_dict, save_path=None):
        """
        显示多相机图像
        image_dict: 包含多个相机图像的字典
        """
        num_cameras = len(self.camera_names)
        cols = 2
        rows = (num_cameras + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle('多相机图像显示', fontsize=16)
        
        if num_cameras == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, camera_name in enumerate(self.camera_names):
            if camera_name in image_dict:
                image = image_dict[camera_name]
                
                # 处理不同的图像格式
                if isinstance(image, str):  # base64编码
                    image_data = base64.b64decode(image)
                    image = Image.open(io.BytesIO(image_data))
                    image = np.array(image)
                
                # 确保图像是RGB格式
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    axes[i].imshow(image)
                else:
                    axes[i].imshow(image, cmap='gray')
                    
                axes[i].set_title(f'{camera_name}')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No image for\n{camera_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_cameras, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多相机图像已保存到: {save_path}")
        
        plt.show()


def visualize_openvla_prediction(image, action, camera_names=None):
    """
    可视化OpenVLA预测结果
    """
    if camera_names is None:
        camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_d455']
    
    # 创建可视化器
    action_viz = ActionVisualizer()
    
    # 如果action是字典格式，提取数值
    if isinstance(action, dict):
        # 假设action包含位置和夹爪信息
        if 'action' in action:
            action_array = action['action']
        else:
            # 尝试从字典中提取数值
            action_array = []
            for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']:
                if key in action:
                    action_array.append(action[key])
            action_array = np.array(action_array)
    else:
        action_array = np.array(action)
    
    # 实时可视化
    action_viz.plot_real_time(action_array, image)
    
    return action_viz


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    visualizer = ActionVisualizer()
    
    # 模拟一系列动作
    for i in range(50):
        # 生成示例动作 [x, y, z, roll, pitch, yaw, gripper]
        action = [
            0.5 + 0.1 * np.sin(i * 0.1),  # x
            0.3 + 0.1 * np.cos(i * 0.1),  # y  
            0.2 + 0.05 * np.sin(i * 0.2), # z
            0.1 * np.sin(i * 0.05),       # roll
            0.1 * np.cos(i * 0.05),       # pitch
            i * 0.02,                     # yaw
            0.5 + 0.5 * np.sin(i * 0.3)   # gripper
        ]
        visualizer.add_action(action)
    
    # 绘制轨迹
    visualizer.plot_action_trajectory('action_trajectory.png')