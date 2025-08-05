import cv2
import torch
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        初始化YOLOv11s-Pose人体姿态检测器（开源版本，仅支持PyTorch模型）
        
        Args:
            model_path (str): YOLOv11s-Pose模型路径（仅支持.pt模型）
            conf_threshold (float): 置信度阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # 开源版本仅支持PyTorch模型
        if not model_path.endswith('.pt'):
            raise ValueError("开源版本仅支持.pt模型")
        
        self.model = YOLO(model_path)
        print(f"成功加载PyTorch模型: {model_path}")
    
    def detect(self, frame):
        """
        检测人体姿态关键点
        
        Args:
            frame (np.ndarray): 输入视频帧
            
        Returns:
            list: 检测到的人体边界框列表 [(x1, y1, x2, y2), ...]
        """
        return self._detect_pytorch(frame)
    
    def _detect_pytorch(self, frame):
        """使用PyTorch模型进行检测"""
        results = self.model(frame, verbose=False)
        boxes = []
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if box.conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return boxes