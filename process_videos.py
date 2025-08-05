#!/usr/bin/env python3
"""
视频人体打码处理脚本（开源版本） - 使用YOLOv11-Pose模型对视频中所有人体进行检测和带羽化的高斯模糊打码
"""

import os
import sys
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用相对导入
from processor.video_processor import VideoProcessor
from config import load_config

def main():
    parser = argparse.ArgumentParser(description='Video Human Body Mosaic Processor with Feathered Blur (Open Source Version)')
    parser.add_argument('--input', '-i', required=True, help='Input video file or directory')
    parser.add_argument('--pose-model', default=None, help='YOLOv11s-Pose model path (only .pt models supported)')
    parser.add_argument('--pose-conf', type=float, default=None, help='Pose detection confidence threshold')
    parser.add_argument('--output', '-o', required=True, help='Output video file or directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show detection boxes')
    parser.add_argument('--blur-kernel', type=int, default=None, help='Gaussian blur kernel size')
    parser.add_argument('--blur-sigma', type=float, default=None, help='Gaussian blur sigma X value')
    parser.add_argument('--blur-feather', type=int, default=None, help='Feather strength for blur edges')
    parser.add_argument('--frame-extension-pre', type=int, default=None, help='Frame pre-extension range')
    parser.add_argument('--frame-extension-post', type=int, default=None, help='Frame post-extension range')
    parser.add_argument('--keyframe-interval', type=int, default=None, help='Keyframe interval')
    parser.add_argument('--keyframe-pre', type=int, default=None, help='Keyframe pre-extension frames')
    parser.add_argument('--keyframe-post', type=int, default=None, help='Keyframe post-extension frames')
    parser.add_argument('--codec', choices=['h264', 'h265'], default=None, help='Video codec')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 使用命令行参数覆盖配置
    pose_model = args.pose_model if args.pose_model is not None else config['pose_model']
    pose_conf = args.pose_conf if args.pose_conf is not None else config['pose_conf_threshold']
    frame_extension_pre = args.frame_extension_pre if args.frame_extension_pre is not None else config['blur_pre_frames']
    frame_extension_post = args.frame_extension_post if args.frame_extension_post is not None else config['blur_post_frames']
    blur_kernel_size = args.blur_kernel if args.blur_kernel is not None else config['blur_kernel_size']
    blur_sigma_x = args.blur_sigma if args.blur_sigma is not None else config['blur_sigma_x']
    blur_feather_strength = args.blur_feather if args.blur_feather is not None else config['blur_feather_strength']
    keyframe_interval = args.keyframe_interval if args.keyframe_interval is not None else config['keyframe_interval']
    keyframe_pre_frames = args.keyframe_pre if args.keyframe_pre is not None else config['keyframe_extrapolate_pre']
    keyframe_post_frames = args.keyframe_post if args.keyframe_post is not None else config['keyframe_extrapolate_post']
    video_codec = args.codec if args.codec is not None else config['video_codec']
    
    debug_mode = args.debug
    
    # 检查模型文件是否存在
    if not os.path.exists(pose_model):
        print(f"Error: Pose model file not found at {pose_model}")
        sys.exit(1)
    
    # 创建处理器实例，传递配置参数
    processor = VideoProcessor(
        pose_model_path=pose_model,
        pose_conf_threshold=pose_conf,
        blur_kernel_size=blur_kernel_size,
        blur_sigma_x=blur_sigma_x,
        blur_feather_strength=blur_feather_strength,
        debug_mode=debug_mode,
        keyframe_interval=keyframe_interval,
        video_codec=video_codec
    )
    
    # 处理单个文件
    if os.path.isfile(args.input):
        if not args.output.endswith('.mp4'):
            print("Error: Output file must have .mp4 extension")
            sys.exit(1)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        
        print(f"Processing video: {args.input}")
        processor.process_video(args.input, args.output)
        print(f"Processed video saved to: {args.output}")
    
    # 处理目录
    elif os.path.isdir(args.input):
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        # 处理目录中的所有视频文件
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, f"mosaic_{filename}")
                
                print(f"Processing video: {input_path}")
                processor.process_video(input_path, output_path)
                print(f"Processed video saved to: {output_path}")
    
    else:
        print(f"Error: Input path {args.input} does not exist")
        sys.exit(1)

if __name__ == "__main__":
    main()