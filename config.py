import os

# 开源版本配置
DEFAULT_CONFIG = {
    # 模型路径 - 仅支持.pt模型
    'pose_model': 'models/yolo11s-pose.pt',
    
    # 置信度阈值
    'pose_conf_threshold': 0.5,
    
    # 帧扩展范围（用于打码范围扩展）
    'blur_pre_frames': 5,   # 打码前扩展帧数
    'blur_post_frames': 5,  # 打码后扩展帧数
    
    # 高斯模糊参数 - 仅使用羽化版本
    'blur_kernel_size': 61,      # 增大高斯核大小以获得更强的模糊效果
    'blur_sigma_x': 0,            # X方向标准差，0表示自动计算
    'blur_feather_strength': 20,  # 羽化强度，值越大羽化效果越明显
    
    # 关键帧间隔
    'keyframe_interval': 5,
    
    # 关键帧外推参数（用于关键帧智能分布）
    'keyframe_extrapolate_pre': 6,    # 关键帧向前外推帧数（在检测区间开始前增加关键帧检测）
    'keyframe_extrapolate_post': 8,   # 关键帧向后外推帧数（在检测区间结束后增加关键帧检测）
    
    # 视频编码选项：'h264' 或 'h265'
    'video_codec': 'h264',
}

def load_config(config_path=None):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    config = DEFAULT_CONFIG.copy()
    
    # 如果提供了配置文件路径，则加载配置文件
    if config_path and os.path.exists(config_path):
        # 这里可以实现从文件加载配置的逻辑
        # 为了简化，我们暂时不实现文件加载
        pass
    
    return config