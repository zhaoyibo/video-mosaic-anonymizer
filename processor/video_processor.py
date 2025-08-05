import cv2
import os
import sys
import numpy as np
import time

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 使用相对导入
from detector.pose_detector import PoseDetector
from mosaic.feathered_blur import FeatheredBlur
from utils.video_utils import expand_boxes

class VideoProcessor:
    def __init__(self, pose_model_path, 
                 pose_conf_threshold=0.5,
                 blur_kernel_size=61, blur_sigma_x=0,
                 blur_feather_strength=20,
                 debug_mode=False, keyframe_interval=5,
                 video_codec='h264'):
        """
        初始化视频处理器（开源版本）

        Args:
            pose_model_path (str): 人体姿态检测模型路径（仅支持.pt模型）
            pose_conf_threshold (float): 人体检测置信度阈值
            blur_kernel_size (int): 高斯模糊核大小
            blur_sigma_x (float): 高斯模糊X方向标准差
            blur_feather_strength (int): 羽化强度，值越大羽化效果越明显
            debug_mode (bool): 是否启用调试模式
            keyframe_interval (int): 关键帧间隔
            video_codec (str): 视频编码选项 ('h264' 或 'h265')
        """
        # 仅支持PyTorch模型（.pt）
        if not pose_model_path.endswith('.pt'):
            raise ValueError("开源版本仅支持.pt模型")
        
        # PyTorch模型
        self.pose_detector = PoseDetector(pose_model_path, pose_conf_threshold)
        print("使用PyTorch检测器")
        
        self.gaussian_blur = None  # 不再使用普通高斯模糊
        # 默认使用羽化模糊处理器，提供更好的视觉效果
        self.feathered_blur = FeatheredBlur(blur_kernel_size, blur_sigma_x, feather_strength=blur_feather_strength)
        self.debug_mode = debug_mode
        self.keyframe_interval = keyframe_interval
        self.video_codec = video_codec
        print(f"初始化视频处理器，请求的编码格式: {self.video_codec}")
        
        # 创建调试日志文件
        self.debug_log = open("debug_log.txt", "w")
        # 记录真正应用了打码的帧范围
        self.actual_blurred_frames = []

    def interpolate_boxes(self, frame_detections, total_frames):
        """
        对检测框进行插值以填补漏检帧和减少跳跃

        Args:
            frame_detections (list): 每帧的检测结果
            total_frames (int): 总帧数

        Returns:
            list: 插值后的检测结果
        """
        if not frame_detections:
            return frame_detections

        # 创建插值后的结果列表
        interpolated_detections = [None] * total_frames

        # 找到所有有检测结果的帧
        detected_frames = []
        for i, detection in enumerate(frame_detections):
            if detection is not None:
                detected_frames.append(i)

        if len(detected_frames) <= 1:
            return frame_detections

        # 识别人体出现的连续时间段
        segments = []
        if detected_frames:
            start = detected_frames[0]
            end = detected_frames[0]

            for i in range(1, len(detected_frames)):
                if detected_frames[i] == end + 1:
                    end = detected_frames[i]
                else:
                    segments.append((start, end))
                    start = detected_frames[i]
                    end = detected_frames[i]

            segments.append((start, end))

            # 对每个时间段分别进行插值处理
            for segment_start, segment_end in segments:
                # 获取时间段内的检测结果
                segment_detections = {}
                for frame_idx in range(segment_start, segment_end + 1):
                    if frame_detections[frame_idx] is not None:
                        segment_detections[frame_idx] = frame_detections[frame_idx]

                # 对时间段内的每一帧进行插值
                for frame_idx in range(segment_start, segment_end + 1):
                    if frame_idx in segment_detections:
                        # 关键帧，直接使用检测结果
                        interpolated_detections[frame_idx] = segment_detections[frame_idx]
                    else:
                        # 非关键帧，进行插值
                        # 找到前一个和后一个关键帧
                        prev_frame = None
                        next_frame = None

                        # 在当前段内查找前一个关键帧
                        for detected_frame in reversed(list(segment_detections.keys())):
                            if detected_frame < frame_idx:
                                prev_frame = detected_frame
                                break

                        # 在当前段内查找后一个关键帧
                        for detected_frame in segment_detections.keys():
                            if detected_frame > frame_idx:
                                next_frame = detected_frame
                                break

                        # 如果找不到前一个或后一个关键帧，则使用最近的关键帧
                        if prev_frame is None and next_frame is not None:
                            interpolated_detections[frame_idx] = segment_detections[next_frame]
                        elif next_frame is None and prev_frame is not None:
                            interpolated_detections[frame_idx] = segment_detections[prev_frame]
                        elif prev_frame is not None and next_frame is not None:
                            # 进行线性插值
                            prev_detection = segment_detections[prev_frame]
                            next_detection = segment_detections[next_frame]

                            # 矩形框插值
                            prev_boxes = prev_detection
                            next_boxes = next_detection

                            # 对每个边界框进行插值
                            interpolated_boxes = []
                            for i in range(min(len(prev_boxes), len(next_boxes))):
                                x1_prev, y1_prev, x2_prev, y2_prev = prev_boxes[i]
                                x1_next, y1_next, x2_next, y2_next = next_boxes[i]

                                # 线性插值
                                x1_interp = int(x1_prev + (x1_next - x1_prev) * (frame_idx - prev_frame) / (next_frame - prev_frame))
                                y1_interp = int(y1_prev + (y1_next - y1_prev) * (frame_idx - prev_frame) / (next_frame - prev_frame))
                                x2_interp = int(x2_prev + (x2_next - x2_prev) * (frame_idx - prev_frame) / (next_frame - prev_frame))
                                y2_interp = int(y2_prev + (y2_next - y2_prev) * (frame_idx - prev_frame) / (next_frame - prev_frame))

                                interpolated_boxes.append((x1_interp, y1_interp, x2_interp, y2_interp))

                            interpolated_detections[frame_idx] = interpolated_boxes
                        else:
                            # 如果没有任何检测结果，使用空结果
                            interpolated_detections[frame_idx] = []

            # 对于没有检测到人体的帧，使用空结果
            for frame_idx in range(total_frames):
                if interpolated_detections[frame_idx] is None:
                    interpolated_detections[frame_idx] = []

        return interpolated_detections

    def process_video(self, input_path, output_path):
        """
        处理视频文件，仅保留根据人体矩形框打码的方法

        Args:
            input_path (str): 输入视频文件路径
            output_path (str): 输出视频文件路径
        """
        print("进入process_video方法")
        start_time = time.time()

        # 打开视频文件
        cap = cv2.VideoCapture(input_path)

        # 获取输入视频属性
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算视频时长
        duration = input_total_frames / input_fps if input_fps > 0 else 0

        # 设置输出视频属性
        output_fps = input_fps
        output_width = input_width
        output_height = input_height

        # 选择编码器 - 使用OpenCV支持的编码器
        intermediate_codec = None
        
        # 从环境变量获取H.264编码器列表，如果未设置则使用默认值
        h264_codes_str = os.getenv('H264_ENCODER_CODES', 'X264,H264,avc1,XVID')
        h264_codes = [code.strip() for code in h264_codes_str.split(',')]
        
        if self.video_codec == 'h265':
            # 首先尝试H.264，然后在后处理中转换为H.265
            # 这样可以避免OpenCV的编码器兼容性问题
            # 尝试多种H.264编码器标识符，提高兼容性
            h264_success = False
            for code in h264_codes:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*code)
                    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
                    if out.isOpened():
                        intermediate_codec = 'h264'
                        print(f"使用H.264编码器: {code}")
                        h264_success = True
                        break
                    else:
                        print(f"H.264编码器 {code} 无法打开")
                except Exception as e:
                    print(f"H.264编码器 {code} 不可用: {e}")
            
            if not h264_success:
                print("所有H.264编码器都不可用，使用默认MPEG-4编码")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
                intermediate_codec = 'mpeg4'
        else:
            # 使用H.264编码
            # 尝试多种H.264编码器标识符，提高兼容性
            h264_success = False
            for code in h264_codes:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*code)
                    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
                    if out.isOpened():
                        intermediate_codec = 'h264'
                        print(f"使用H.264编码器: {code}")
                        h264_success = True
                        break
                    else:
                        print(f"H.264编码器 {code} 无法打开")
                except Exception as e:
                    print(f"H.264编码器 {code} 不可用: {e}")
            
            if not h264_success:
                # 如果H.264不可用，回退到默认编码
                print("警告：H.264编码不可用，使用默认MPEG-4编码")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
                intermediate_codec = 'mpeg4'

        # 存储中间编码信息供后续使用
        self.intermediate_codec = intermediate_codec
        print(f"中间编码: {intermediate_codec}")

        # 用于记录检测到人体的帧
        pose_frames = []

        # 存储所有帧的检测结果，用于后续处理
        frame_detections = []

        # 用于记录人体第一次出现的帧
        first_appearance_frames = []

        # 用于记录检测到的段（连续的检测帧区间）
        detected_segments = []
        current_segment_start = None

        frame_count = 0
        # 初始化检测结果变量
        pose_boxes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 重置检测结果变量
            pose_boxes = []

            # 完成一个段的记录
            has_detection = len(pose_boxes) > 0

            if current_segment_start is not None and (not has_detection or (pose_frames and frame_count - pose_frames[-1] > 1)):
                detected_segments.append((current_segment_start, pose_frames[-1]))
                current_segment_start = None

            # 开始一个新段
            if has_detection and current_segment_start is None:
                current_segment_start = frame_count

            # 每30帧打印一次处理进度
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}")

            # 只在关键帧进行检测
            is_keyframe = (frame_count % self.keyframe_interval == 0)

            # 对于检测到人体的片段的边界关键帧，需要额外检测以确保完整性
            is_boundary_keyframe = False
            if pose_frames:  # 如果已经有检测到的帧
                last_pose_frame = pose_frames[-1]
                # 如果当前帧接近上一个检测到人体的帧，则也作为关键帧处理
                if frame_count - last_pose_frame <= self.keyframe_interval * 2:
                    is_boundary_keyframe = True

            if is_keyframe or is_boundary_keyframe:
                # 检测人体（仅使用矩形框方式）
                pose_boxes = self.pose_detector.detect(frame)

            # 保存当前帧的检测结果（关键帧保存实际结果，非关键帧保存None）
            if is_keyframe or is_boundary_keyframe:
                frame_detections.append(pose_boxes)
            else:
                frame_detections.append(None)  # 非关键帧标记为None，后续进行插值

            # 记录检测到人体的关键帧
            if is_keyframe or is_boundary_keyframe:
                if len(pose_boxes) > 0:
                    pose_frames.append(frame_count)
                    # 如果这是第一次检测到人体，则记录下来
                    if len(pose_frames) == 1:
                        first_appearance_frames.append(frame_count)
                        current_segment_start = frame_count

            frame_count += 1

        # 完成最后一个段的记录
        if current_segment_start is not None and len(pose_boxes) > 0:
            detected_segments.append((current_segment_start, pose_frames[-1]))

        # 对检测结果进行插值以减少帧间跳跃和漏检
        interpolated_frame_detections = self.interpolate_boxes(frame_detections, frame_count)

        # 重新处理视频，应用扩展后的帧范围
        cap.release()
        cap = cv2.VideoCapture(input_path)

        # 重置帧计数器
        frame_count = 0
        # 重置实际打码帧记录
        self.actual_blurred_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检查当前帧是否在检测范围内（简化处理，不使用帧扩展）
            # 直接使用插值后的检测结果
            detection_result = interpolated_frame_detections[frame_count] if frame_count < len(interpolated_frame_detections) else []

            # 添加调试信息到文件
            if frame_count % 30 == 0:  # 每30帧记录一次调试信息
                self.debug_log.write(f"Frame {frame_count}: detection_result={detection_result}\n")

            # 确保检测结果不为None
            if detection_result is not None:
                self.debug_log.write(f"Frame {frame_count}: Detection result is not None\n")
                # 检查检测结果是否有效
                has_detection = len(detection_result) > 0
                self.debug_log.write(f"Frame {frame_count}: Has detection: {has_detection}\n")

                if has_detection:
                    # 处理人体矩形框检测
                    pose_boxes = detection_result
                    self.debug_log.write(f"Frame {frame_count}: Processing with rectangle method, boxes: {pose_boxes}\n")  # 添加调试信息
                    
                    # 扩展边界框，使用更大的扩展比例确保完全覆盖人体
                    expanded_boxes = expand_boxes(pose_boxes, expand_ratio=0.2)

                    # 调试模式下绘制检测框
                    if self.debug_mode:
                        # 先应用羽化模糊效果
                        if expanded_boxes:
                            self.debug_log.write(f"Frame {frame_count}: Applying feathered blur to boxes: {expanded_boxes}\n")  # 添加调试信息
                            blurred_frame = self.feathered_blur.apply(frame.copy(), expanded_boxes)
                            # 记录实际打码的帧
                            if frame_count not in self.actual_blurred_frames: 
                                self.actual_blurred_frames.append(frame_count)
                        else:
                            blurred_frame = frame.copy()

                        # 在模糊后的帧上绘制检测框
                        # 绘制原始检测框（蓝色）
                        for (x1, y1, x2, y2) in pose_boxes:
                            cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # 绘制扩展后的检测框（绿色）
                        for (x1, y1, x2, y2) in expanded_boxes:
                            cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 在右上角显示帧数
                        cv2.putText(blurred_frame, f"Frame: {frame_count}", (blurred_frame.shape[1] - 200, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                        processed_frame = blurred_frame
                    else:
                        # 应用羽化模糊
                        if expanded_boxes:  # 添加检查确保有边界框
                            self.debug_log.write(f"Frame {frame_count}: Applying feathered blur to boxes: {expanded_boxes}\n")  # 添加调试信息
                            processed_frame = self.feathered_blur.apply(frame, expanded_boxes)
                            # 记录实际打码的帧
                            if frame_count not in self.actual_blurred_frames: 
                                self.actual_blurred_frames.append(frame_count)
                        else:
                            self.debug_log.write(f"Frame {frame_count}: No boxes to blur\n")  # 添加调试信息
                            processed_frame = frame
                else:
                    # 如果没有检测到人体，直接使用原帧
                    processed_frame = frame
            else:
                # 如果检测结果为None，直接使用原帧
                processed_frame = frame

            # 如果处于调试模式，在所有帧上都显示帧数
            if self.debug_mode:
                debug_frame = processed_frame.copy()
                # 在右上角显示帧数
                cv2.putText(debug_frame, f"Frame: {frame_count}", (debug_frame.shape[1] - 200, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                processed_frame = debug_frame

            # 写入处理后的帧
            out.write(processed_frame)

            frame_count += 1

        # 释放资源
        cap.release()
        out.release()

        # 写入处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        self.debug_log.write(f"Processing time: {processing_time:.2f} seconds\n")

        # 格式化帧范围信息
        def format_frame_ranges(frames):
            if not frames:
                return []

            ranges = []
            start = frames[0]
            end = frames[0]

            for i in range(1, len(frames)):
                if frames[i] == end + 1:
                    end = frames[i]
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = frames[i]
                    end = frames[i]

            # 添加最后一个范围
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")

            return ranges

        pose_ranges = format_frame_ranges(pose_frames)
        actual_blurred_ranges = format_frame_ranges(self.actual_blurred_frames)

        self.debug_log.write(f"Video processing completed.\n")
        self.debug_log.write(f"  Total frames: {frame_count}\n")
        self.debug_log.write(f"  Video duration: {duration:.2f} seconds\n")
        self.debug_log.write(f"  Processing time: {processing_time:.2f} seconds\n")
        self.debug_log.write(f"  Pose detection frames: {pose_ranges}\n")
        self.debug_log.write(f"  Actual blurred frames: {actual_blurred_ranges}\n")
        self.debug_log.write(f"  Blur kernel size: {self.feathered_blur.kernel_size}\n")
        self.debug_log.write(f"  Blur sigma X: {self.feathered_blur.sigma_x}\n")

        print(f"Video processing completed.")
        print(f"  Total frames: {frame_count}")
        print(f"  Video duration: {duration:.2f} seconds")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Pose detection frames: {pose_ranges}")
        print(f"  Actual blurred frames: {actual_blurred_ranges}")
        print(f"  Blur kernel size: {self.feathered_blur.kernel_size}")
        print(f"  Blur sigma X: {self.feathered_blur.sigma_x}")
        print(f"视频处理完成，开始后处理，video_codec={self.video_codec}")
        print("准备执行编码转换逻辑")

        # 如果需要转换编码，则进行转换
        print(f"检查是否需要编码转换: video_codec={self.video_codec}")
        if self.video_codec in ['h264', 'h265']:
            print(f"目标编码: {self.video_codec}")
            # 检查当前编码
            try:
                import subprocess
                import json
                probe_result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', '-select_streams', 'v:0', output_path
                ], capture_output=True, text=True, timeout=30)
                
                if probe_result.returncode == 0:
                    info = json.loads(probe_result.stdout)
                    if 'streams' in info and len(info['streams']) > 0:
                        stream = info['streams'][0]
                        current_codec = stream.get('codec_name', '')
                        # 确定目标编码名称：H.265的目标编码名称是'hevc'
                        target_codec_name = 'hevc' if self.video_codec == 'h265' else 'h264'
                        
                        print(f"当前编码: {current_codec}, 目标编码: {target_codec_name}")
                        # 如果当前编码与目标编码不一致，则进行转换
                        if current_codec != target_codec_name:
                            print(f"当前编码 {current_codec} 与目标编码 {target_codec_name} 不一致，需要转换")
                            success = self._convert_to_target_codec(output_path, self.video_codec)
                            if success:
                                print(f"编码转换完成: {self.video_codec}")
                            else:
                                print("编码转换失败")
                        else:
                            print(f"视频已使用 {current_codec} 编码")
                else:
                    print(f"无法检查视频编码信息: {probe_result.stderr}")
            except Exception as e:
                print(f"检查视频编码时出错: {e}")
        else:
            print(f"无需编码转换: {self.video_codec}")
        
        # 关闭调试日志文件
        self.debug_log.close()

    def _convert_to_target_codec(self, input_path, target_codec, target_bitrate=None):
        """
        将视频转换为目标编码格式

        Args:
            input_path (str): 输入视频路径
            target_codec (str): 目标编码格式 ('h264' 或 'h265')
            target_bitrate (int, optional): 目标比特率

        Returns:
            bool: 转换是否成功
        """
        try:
            import subprocess
            import tempfile
            import shutil
            import os

            # 检查输入文件是否存在
            if not os.path.exists(input_path):
                print(f"输入文件不存在: {input_path}")
                return False

            # 创建临时文件
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_output.close()

            # 选择编码器
            if target_codec == 'h265':
                encoder = 'libx265'
                profile = 'main'
            else:  # h264
                encoder = 'libx264'
                profile = 'main'

            # 构建FFmpeg命令
            encode_params = [
                'ffmpeg', '-i', input_path,
                '-c:v', encoder,
                '-profile:v', profile,
            ]

            # 设置比特率
            if target_bitrate:
                encode_params.extend(['-b:v', str(target_bitrate)])
                print(f"设置目标视频比特率: {target_bitrate} bps")
            else:
                # 使用默认质量设置
                encode_params.extend(['-crf', '23'])

            # 添加音频和容器参数
            encode_params.extend([
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y', temp_output.name
            ])

            print(f"FFmpeg转换命令: {' '.join(encode_params)}")

            # 执行转换
            encode_result = subprocess.run(
                encode_params,
                capture_output=True, text=True, timeout=120
            )

            if encode_result.returncode == 0:
                # 替换原文件
                shutil.move(temp_output.name, input_path)
                print(f"视频已转换编码为{target_codec.upper()}")
                return True
            else:
                # 如果ffmpeg失败，删除临时文件
                os.unlink(temp_output.name)
                print(f"视频转换失败: {encode_result.stderr}")
                return False

        except Exception as e:
            print(f"视频转换过程出错: {e}")
            # 清理临时文件
            if 'temp_output' in locals():
                try:
                    os.unlink(temp_output.name)
                except:
                    pass
            return False