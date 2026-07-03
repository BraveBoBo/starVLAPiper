import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from visualize_actions import ActionVisualizer, MultiCameraVisualizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_visualization_data(file_path='visualization_data.pkl'):
    """
    加载保存的可视化数据
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载数据文件: {file_path}")
        print(f"包含 {len(data['actions'])} 个动作数据点")
        return data
    except FileNotFoundError:
        print(f"未找到数据文件: {file_path}")
        return None
    except Exception as e:
        print(f"加载数据文件出错: {e}")
        return None

def analyze_action_patterns(data):
    """
    分析动作模式
    """
    if data is None or len(data['actions']) == 0:
        print("无有效数据进行分析")
        return
    
    actions = np.array(data['actions'])
    timestamps = np.array(data['timestamps'])
    
    print(f"动作数据分析:")
    print(f"总步数: {len(actions)}")
    print(f"动作维度: {actions.shape[1] if len(actions.shape) > 1 else 1}")
    
    # 统计信息
    if len(actions.shape) > 1:
        action_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        fig.suptitle('动作数据统计分析', fontsize=16)
        
        for i in range(min(actions.shape[1], 7)):
            # 时间序列
            row, col = i // 4, i % 4
            axes[row, col].plot(actions[:, i], linewidth=2)
            axes[row, col].set_title(f'{action_labels[i]} 时间序列')
            axes[row, col].set_xlabel('步数')
            axes[row, col].grid(True)
            
            # 添加统计信息
            mean_val = np.mean(actions[:, i])
            std_val = np.std(actions[:, i])
            min_val = np.min(actions[:, i])
            max_val = np.max(actions[:, i])
            
            axes[row, col].text(0.02, 0.98, 
                f'均值: {mean_val:.3f}\n标准差: {std_val:.3f}\n最小值: {min_val:.3f}\n最大值: {max_val:.3f}',
                transform=axes[row, col].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(actions.shape[1], 8):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('action_statistics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 动作变化率分析
        if len(actions) > 1:
            action_changes = np.diff(actions, axis=0)
            change_norms = np.linalg.norm(action_changes, axis=1)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(change_norms, linewidth=2)
            plt.title('动作变化幅度')
            plt.xlabel('步数')
            plt.ylabel('变化幅度')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.hist(change_norms, bins=30, alpha=0.7, edgecolor='black')
            plt.title('动作变化幅度分布')
            plt.xlabel('变化幅度')
            plt.ylabel('频次')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('action_change_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

def create_action_video(data, output_path='action_video.mp4', fps=10):
    """
    创建动作轨迹动画视频
    """
    if data is None or len(data['actions']) == 0:
        print("无有效数据创建视频")
        return
    
    actions = np.array(data['actions'])
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': '3d'})
    
    def animate(frame):
        ax.clear()
        
        # 绘制到当前帧的轨迹
        current_actions = actions[:frame+1]
        
        ax.plot(current_actions[:, 0], current_actions[:, 1], current_actions[:, 2], 
                'b-', linewidth=2, alpha=0.7)
        
        # 当前位置
        if frame < len(actions):
            ax.scatter(actions[frame, 0], actions[frame, 1], actions[frame, 2], 
                      c='red', s=100, alpha=1.0)
        
        # 起点和终点
        ax.scatter(actions[0, 0], actions[0, 1], actions[0, 2], 
                  c='green', s=100, marker='o', label='起点')
        if frame == len(actions) - 1:
            ax.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2], 
                      c='red', s=100, marker='s', label='终点')
        
        ax.set_xlabel('X位置 (m)')
        ax.set_ylabel('Y位置 (m)')
        ax.set_zlabel('Z位置 (m)')
        ax.set_title(f'机器人轨迹动画 - 步数: {frame+1}/{len(actions)}')
        ax.legend()
        
        # 设置固定的视角范围
        x_range = [np.min(actions[:, 0]), np.max(actions[:, 0])]
        y_range = [np.min(actions[:, 1]), np.max(actions[:, 1])]
        z_range = [np.min(actions[:, 2]), np.max(actions[:, 2])]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
    
    try:
        from matplotlib.animation import FuncAnimation
        
        anim = FuncAnimation(fig, animate, frames=len(actions), interval=100, repeat=True)
        anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"轨迹动画已保存到: {output_path}")
    except Exception as e:
        print(f"创建动画视频失败: {e}")
        print("请确保安装了ffmpeg")

def create_image_mosaic(data, save_path='image_mosaic.png', grid_size=(4, 4)):
    """
    创建图像拼贴
    """
    if data is None or len(data['images']) == 0:
        print("无图像数据")
        return
    
    images = data['images']
    rows, cols = grid_size
    total_images = rows * cols
    
    # 选择均匀分布的图像
    indices = np.linspace(0, len(images)-1, total_images, dtype=int)
    
    # 创建拼贴
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle(f'图像时间序列拼贴 (总共{len(images)}张)', fontsize=16)
    
    for i, idx in enumerate(indices):
        row, col = i // cols, i % cols
        
        image = images[idx]
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                axes[row, col].imshow(image)
            else:
                axes[row, col].imshow(image, cmap='gray')
        
        axes[row, col].set_title(f'步数 {idx+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图像拼贴已保存到: {save_path}")

def performance_analysis(data):
    """
    性能分析
    """
    if data is None or len(data['timestamps']) == 0:
        print("无时间戳数据进行性能分析")
        return
    
    timestamps = np.array(data['timestamps'])
    
    if len(timestamps) > 1:
        # 计算执行间隔
        intervals = np.diff(timestamps)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(intervals, linewidth=2)
        plt.title('执行时间间隔')
        plt.xlabel('步数')
        plt.ylabel('时间间隔 (秒)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.hist(intervals, bins=30, alpha=0.7, edgecolor='black')
        plt.title('时间间隔分布')
        plt.xlabel('时间间隔 (秒)')
        plt.ylabel('频次')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        freq = 1.0 / intervals
        plt.plot(freq, linewidth=2)
        plt.title('执行频率')
        plt.xlabel('步数')
        plt.ylabel('频率 (Hz)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        cumulative_time = timestamps - timestamps[0]
        plt.plot(cumulative_time, linewidth=2)
        plt.title('累积执行时间')
        plt.xlabel('步数')
        plt.ylabel('时间 (秒)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"性能统计:")
        print(f"平均执行间隔: {np.mean(intervals):.3f} 秒")
        print(f"标准差: {np.std(intervals):.3f} 秒")
        print(f"平均执行频率: {np.mean(freq):.2f} Hz")
        print(f"总执行时间: {cumulative_time[-1]:.2f} 秒")

def main():
    """
    主分析函数
    """
    print("开始可视化数据分析...")
    
    # 加载数据
    data = load_visualization_data()
    
    if data is None:
        print("请确保运行openvla_cus.py后生成了visualization_data.pkl文件")
        return
    
    # 创建输出目录
    output_dir = 'visualization_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 切换到输出目录
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # 执行各种分析
        print("\\n1. 动作模式分析...")
        analyze_action_patterns(data)
        
        print("\\n2. 创建图像拼贴...")
        create_image_mosaic(data)
        
        print("\\n3. 性能分析...")
        performance_analysis(data)
        
        print("\\n4. 创建轨迹动画...")
        create_action_video(data)
        
        # 使用ActionVisualizer创建详细轨迹图
        print("\\n5. 生成详细轨迹图...")
        visualizer = ActionVisualizer()
        for action in data['actions']:
            visualizer.add_action(action)
        visualizer.plot_action_trajectory('detailed_trajectory.png')
        
        print("\\n所有分析完成！结果保存在 visualization_analysis 目录中")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
    finally:
        # 返回原目录
        os.chdir(original_dir)

if __name__ == "__main__":
    main()