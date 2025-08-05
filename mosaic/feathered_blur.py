import cv2
import numpy as np

class FeatheredBlur:
    def __init__(self, kernel_size=61, sigma_x=0, feather_strength=15):
        """
        初始化羽化模糊处理器
        
        Args:
            kernel_size (int): 高斯核大小，必须是正奇数
            sigma_x (float): X方向的标准差，0表示自动计算
            feather_strength (int): 羽化强度，值越大羽化效果越明显
        """
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.feather_strength = feather_strength
    
    def create_feathered_mask(self, frame_shape, boxes, feather_strength=None):
        """
        创建带羽化边缘的mask
        
        Args:
            frame_shape (tuple): 帧的形状 (height, width, channels)
            boxes (list): 需要模糊的区域边界框列表 [(x1, y1, x2, y2), ...]
            feather_strength (int): 羽化强度，如果为None则使用实例变量
            
        Returns:
            np.ndarray: 羽化mask (0-1之间的浮点数)
        """
        if feather_strength is None:
            feather_strength = self.feather_strength
            
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        
        for (x1, y1, x2, y2) in boxes:
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x2 > x1 and y2 > y1:
                # 创建矩形区域mask
                mask[y1:y2, x1:x2] = 1.0
                
                # 如果需要羽化且羽化强度大于0
                if feather_strength > 0:
                    # 计算羽化区域的边界
                    fx1 = max(0, x1 - feather_strength)
                    fy1 = max(0, y1 - feather_strength)
                    fx2 = min(width, x2 + feather_strength)
                    fy2 = min(height, y2 + feather_strength)
                    
                    # 在羽化区域内创建渐变
                    for y in range(fy1, fy2):
                        for x in range(fx1, fx2):
                            # 计算到矩形边界的最小距离
                            if x < x1:
                                dist_x = x1 - x
                            elif x > x2:
                                dist_x = x - x2
                            else:
                                dist_x = 0
                                
                            if y < y1:
                                dist_y = y1 - y
                            elif y > y2:
                                dist_y = y - y2
                            else:
                                dist_y = 0
                                
                            # 使用欧几里得距离
                            distance = np.sqrt(dist_x**2 + dist_y**2)
                            
                            # 如果在羽化区域内，计算alpha值
                            if distance < feather_strength:
                                alpha = 1.0 - (distance / feather_strength)
                                # 只更新更大的alpha值（确保内部区域保持1.0）
                                if alpha > mask[y, x]:
                                    mask[y, x] = alpha
        
        return mask
    
    def apply(self, frame, boxes):
        """
        对指定区域应用带羽化的高斯模糊
        
        Args:
            frame (np.ndarray): 输入视频帧
            boxes (list): 需要模糊的区域边界框列表 [(x1, y1, x2, y2), ...]
            
        Returns:
            np.ndarray: 处理后的视频帧
        """
        if not boxes:
            return frame
            
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # 创建羽化mask
        mask = self.create_feathered_mask(frame.shape, boxes)
        
        # 应用高斯模糊到整个图像
        blurred_frame = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), self.sigma_x)
        blurred_frame = cv2.GaussianBlur(blurred_frame, (self.kernel_size, self.kernel_size), self.sigma_x)
        blurred_frame = cv2.GaussianBlur(blurred_frame, (self.kernel_size, self.kernel_size), self.sigma_x)
        
        # 使用alpha混合实现羽化效果
        mask_3d = np.stack([mask] * 3, axis=-1)
        frame_copy = (mask_3d * blurred_frame + (1 - mask_3d) * frame).astype(np.uint8)
        
        return frame_copy
    
    def apply_with_mask(self, frame, mask):
        """
        对指定mask区域应用带羽化的高斯模糊
        
        Args:
            frame (np.ndarray): 输入视频帧
            mask (np.ndarray): 0-1浮点数mask，表示模糊强度
            
        Returns:
            np.ndarray: 处理后的视频帧
        """
        frame_copy = frame.copy()
        
        # 应用高斯模糊到整个图像
        blurred_frame = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), self.sigma_x)
        blurred_frame = cv2.GaussianBlur(blurred_frame, (self.kernel_size, self.kernel_size), self.sigma_x)
        blurred_frame = cv2.GaussianBlur(blurred_frame, (self.kernel_size, self.kernel_size), self.sigma_x)
        
        # 使用alpha混合实现羽化效果
        mask_3d = np.stack([mask] * 3, axis=-1)
        frame_copy = (mask_3d * blurred_frame + (1 - mask_3d) * frame).astype(np.uint8)
        
        return frame_copy