"""
简单的实时可视化工具
用于快速可视化机器人动作和图像数据
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image
import base64
import io
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleVisualizer:
    def __init__(self):
        """
        简单的可视化器
        """
        self.actions = []
        self.images = []
        self.timestamps = []
        
        # 创建实时显示窗口
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('机器人动作实时可视化', fontsize=16)
        
        plt.ion()  # 开启交互模式
        
    def update_display(self, action, image=None, instruction=""):
        """
        更新显示
        action: 动作数组 [x, y, z, roll, pitch, yaw, gripper]
        image: 图像数据
        instruction: 指令文本
        """
        current_time = time.time()
        
        # 添加数据
        self.actions.append(action)
        self.timestamps.append(current_time)
        if image is not None:
            self.images.append(image)
        
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. 显示当前图像
        if image is not None:
            if isinstance(image, str):  # base64编码
                try:
                    image_data = base64.b64decode(image)
                    image = Image.open(io.BytesIO(image_data))
                    image = np.array(image)
                except:
                    pass
            
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    self.axes[0, 0].imshow(image)
                else:
                    self.axes[0, 0].imshow(image, cmap='gray')
        
        self.axes[0, 0].set_title(f'当前图像\n指令: {instruction}')
        self.axes[0, 0].axis('off')
        
        # 2. 显示当前动作值
        if len(action) >= 7:
            action_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
            
            bars = self.axes[0, 1].barh(range(len(action)), action, color=colors)
            self.axes[0, 1].set_yticks(range(len(action)))
            self.axes[0, 1].set_yticklabels(action_labels)
            self.axes[0, 1].set_xlabel('数值')
            self.axes[0, 1].set_title('当前动作值')
            self.axes[0, 1].grid(True, axis='x')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, action)):
                self.axes[0, 1].text(val + 0.01 * (max(action) - min(action)), i, 
                                   f'{val:.3f}', va='center')
        
        # 3. 显示位置轨迹历史
        if len(self.actions) > 1:
            actions_array = np.array(self.actions)
            times = np.array(self.timestamps) - self.timestamps[0]  # 相对时间
            
            if actions_array.shape[1] >= 3:
                self.axes[1, 0].plot(times, actions_array[:, 0], 'r-', label='X', linewidth=2)
                self.axes[1, 0].plot(times, actions_array[:, 1], 'g-', label='Y', linewidth=2)
                self.axes[1, 0].plot(times, actions_array[:, 2], 'b-', label='Z', linewidth=2)
                
                # 突出显示当前点
                self.axes[1, 0].scatter(times[-1], actions_array[-1, 0], c='red', s=50, zorder=5)
                self.axes[1, 0].scatter(times[-1], actions_array[-1, 1], c='green', s=50, zorder=5)
                self.axes[1, 0].scatter(times[-1], actions_array[-1, 2], c='blue', s=50, zorder=5)
                
            self.axes[1, 0].set_xlabel('时间 (秒)')
            self.axes[1, 0].set_ylabel('位置 (m)')
            self.axes[1, 0].set_title(f'位置轨迹历史 (共{len(self.actions)}步)')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
        
        # 4. 显示3D轨迹
        if len(self.actions) > 1:
            # 移除原有的3D轴并创建新的
            self.axes[1, 1].remove()
            self.axes[1, 1] = self.fig.add_subplot(2, 2, 4, projection='3d')
            
            actions_array = np.array(self.actions)
            if actions_array.shape[1] >= 3:
                self.axes[1, 1].plot(actions_array[:, 0], actions_array[:, 1], actions_array[:, 2], 
                                   'b-', linewidth=2, alpha=0.7)
                
                # 起点和终点
                self.axes[1, 1].scatter(actions_array[0, 0], actions_array[0, 1], actions_array[0, 2], 
                                      c='green', s=100, label='起点')
                self.axes[1, 1].scatter(actions_array[-1, 0], actions_array[-1, 1], actions_array[-1, 2], 
                                      c='red', s=100, label='当前位置')
                
            self.axes[1, 1].set_xlabel('X (m)')
            self.axes[1, 1].set_ylabel('Y (m)')
            self.axes[1, 1].set_zlabel('Z (m)')
            self.axes[1, 1].set_title('3D轨迹')
            self.axes[1, 1].legend()
        
        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新显示
        
    def save_current_plot(self, filename):
        """
        保存当前图像
        """
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {filename}")


def demo_simple_visualizer():
    """
    演示简单可视化器的使用
    """
    visualizer = SimpleVisualizer()
    
    print("开始演示简单可视化器...")
    print("按Ctrl+C停止演示")
    
    try:
        for i in range(200):
            # 生成示例动作数据
            t = i * 0.1
            action = [
                0.5 + 0.2 * np.sin(t),           # x
                0.3 + 0.15 * np.cos(t),          # y  
                0.2 + 0.1 * np.sin(t * 0.5),     # z
                0.2 * np.sin(t * 0.3),           # roll
                0.15 * np.cos(t * 0.3),          # pitch
                t * 0.02,                        # yaw
                0.5 + 0.5 * np.sin(t * 0.7)      # gripper
            ]
            
            # 生成示例图像（彩色噪声）
            image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            
            # 在图像上绘制一些简单图形
            cv2.circle(image, (160 + int(50 * np.sin(t)), 120 + int(30 * np.cos(t))), 
                      20, (0, 255, 0), -1)
            cv2.putText(image, f'Step {i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            # 更新显示
            visualizer.update_display(action, image, f"演示指令 {i+1}")
            
            # 每50步保存一次图像
            if (i + 1) % 50 == 0:
                visualizer.save_current_plot(f'demo_step_{i+1}.png')
            
            time.sleep(0.05)  # 控制演示速度
            
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    
    print("演示结束")


def visualize_from_action_data(actions, images=None, instructions=None):
    """
    从已有的动作数据进行可视化
    actions: 动作数据列表
    images: 图像数据列表（可选）
    instructions: 指令列表（可选）
    """
    visualizer = SimpleVisualizer()
    
    for i, action in enumerate(actions):
        image = images[i] if images and i < len(images) else None
        instruction = instructions[i] if instructions and i < len(instructions) else f"步骤 {i+1}"
        
        visualizer.update_display(action, image, instruction)
        
        # 等待用户输入或自动继续
        if i % 10 == 0:  # 每10步暂停一次
            input(f"按Enter继续... (当前步数: {i+1}/{len(actions)})")


if __name__ == "__main__":
    # 运行演示
    demo_simple_visualizer()