import cv2
import numpy as np
import os
import sys
import subprocess
import json

def merge_boxes(boxes, overlap_threshold=0.3):
    """
    合并重叠的边界框
    
    Args:
        boxes (list): 边界框列表 [(x1, y1, x2, y2), ...]
        overlap_threshold (float): 重叠阈值
        
    Returns:
        list: 合并后的边界框列表
    """
    if not boxes:
        return []
    
    # 按面积排序
    boxes = sorted(boxes, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]), reverse=True)
    
    merged_boxes = []
    
    for box in boxes:
        merged = False
        for i, merged_box in enumerate(merged_boxes):
            # 计算重叠区域
            x1 = max(box[0], merged_box[0])
            y1 = max(box[1], merged_box[1])
            x2 = min(box[2], merged_box[2])
            y2 = min(box[3], merged_box[3])
            
            if x1 < x2 and y1 < y2:
                # 计算重叠面积
                overlap_area = (x2 - x1) * (y2 - y1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                merged_area = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                
                # 计算重叠率
                overlap_ratio = overlap_area / min(box_area, merged_area)
                
                if overlap_ratio > overlap_threshold:
                    # 合并边界框
                    merged_boxes[i] = (
                        min(box[0], merged_box[0]),
                        min(box[1], merged_box[1]),
                        max(box[2], merged_box[2]),
                        max(box[3], merged_box[3])
                    )
                    merged = True
                    break
        
        if not merged:
            merged_boxes.append(box)
    
    return merged_boxes

def expand_boxes(boxes, expand_ratio=0.15):
    """
    扩展边界框以确保完全覆盖目标
    
    Args:
        boxes (list): 边界框列表 [(x1, y1, x2, y2), ...]
        expand_ratio (float): 扩展比例
        
    Returns:
        list: 扩展后的边界框列表
    """
    expanded_boxes = []
    
    for (x1, y1, x2, y2) in boxes:
        width = x2 - x1
        height = y2 - y1
        
        # 增加扩展比例以更好地覆盖人体
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        
        expanded_boxes.append((
            max(0, x1 - expand_x),
            max(0, y1 - expand_y),
            x2 + expand_x,
            y2 + expand_y
        ))
    
    return expanded_boxes

def analyze_video_with_ffprobe(file_path):
    """使用ffprobe检查视频信息"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', file_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            print("\n视频详细信息:")
            
            if 'format' in info:
                format_info = info['format']
                print(f"格式: {format_info.get('format_name', '未知')}")
                print(f"时长: {format_info.get('duration', '未知')} 秒")
                print(f"文件大小: {format_info.get('size', '未知')} 字节")
                print(f"比特率: {format_info.get('bit_rate', '未知')} bps")
                
            if 'streams' in info and len(info['streams']) > 0:
                print("\n视频流信息:")
                for i, stream in enumerate(info['streams']):
                    if stream.get('codec_type') == 'video':
                        print(f"  流 {i}:")
                        print(f"    编码格式: {stream.get('codec_name', '未知')}")
                        print(f"    分辨率: {stream.get('width', '未知')}x{stream.get('height', '未知')}")
                        print(f"    帧率: {stream.get('avg_frame_rate', '未知')}")
                        print(f"    编码类型: {stream.get('codec_type', '未知')}")
            return info
        else:
            print(f"检查视频信息失败: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("未找到ffprobe命令，请安装ffmpeg")
        return None
    except Exception as e:
        print(f"检查视频信息时出错: {e}")
        return None

def check_video_decoding(file_path):
    """使用ffmpeg检查视频是否可以正常解码"""
    try:
        result = subprocess.run([
            'ffmpeg', '-v', 'error', '-i', file_path, 
            '-f', 'null', '-max_muxing_queue_size', '9999', '-'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("\n视频解码测试: 通过")
            return True
        else:
            print(f"\n视频解码测试: 失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("未找到ffmpeg命令，请安装ffmpeg")
        return False
    except Exception as e:
        print(f"视频解码测试时出错: {e}")
        return False