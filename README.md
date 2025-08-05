# 视频人体打码系统

该系统使用YOLOv11-Pose模型对视频中所有人体进行检测，并使用带羽化的高斯模糊进行打码，提供更加自然的视觉效果。

## 功能特点

- 使用YOLOv11-Pose进行人体检测
- 对检测到的所有人体区域进行带羽化的高斯模糊处理，边缘更加柔和自然
- 支持单个视频文件或整个目录的批量处理
- 自动合并重叠检测区域
- 扩展检测区域以确保完全覆盖目标
- 扩展检测帧范围以避免半截人体未被打码
- 提供详细的处理信息输出
- 支持调试模式，可显示检测框用于调优
- 使用关键帧+插值方法减少帧间跳跃和漏检问题
- 输出实际打码的帧数信息
- 支持H.264和H.265视频编码格式
- 仅支持PyTorch (.pt) 模型格式
- 仅提供人体矩形打码功能（不包含面部检测）
- 仅提供带羽化效果的模糊处理

## 环境要求

- Python 3.8+
- CUDA支持（推荐，用于NVIDIA GPU）
- Apple Metal支持（用于Mac M系列芯片）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型文件说明

请注意，模型文件(.pt)不应提交到Git仓库中，因为它们通常很大。请确保在部署环境中提供这些文件。

项目期望的模型文件位于 `models/` 目录下：
- `yolo11s-pose.pt` - YOLOv11s-Pose人体检测模型（推荐）

您可以从以下地址下载模型文件：
- https://github.com/ultralytics/assets/releases

下载后，请将模型文件放置在 `models/` 目录中。

注意：在首次运行项目之前，请确保已下载并放置了模型文件。

## 使用方法

### 处理单个视频文件

```bash
# 使用默认设置处理单个视频文件
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4

# 或者使用启动脚本
./run.sh --input /path/to/input.mp4 --output output/processed_video.mp4
```

### 批量处理目录中的所有视频

```bash
# 批量处理目录中的所有视频文件
python process_videos.py --input /path/to/input_directory --output output

# 或者使用启动脚本
./run.sh --input /path/to/input_directory --output output
```

### 设置羽化强度

```bash
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4 --blur-feather 30
```

羽化强度值越大，边缘越柔和。默认值为20。

### 指定自定义模型路径

```bash
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4 --pose-model /path/to/custom/yolo11s-pose.pt
```

### 启用调试模式查看检测框

```bash
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4 --debug
```

在调试模式下，输出视频会显示：
- 蓝色矩形框：原始检测框
- 绿色矩形框：扩展后的检测框（实际打码区域）
- 视频右上角显示当前帧数

### 设置关键帧间隔

```bash
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4 --keyframe-interval 10
```

### 指定视频编码格式

```bash
# 使用H.264编码（默认）
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4 --codec h264

# 使用H.265编码（文件更小，但需要更长的处理时间）
python process_videos.py --input /path/to/input.mp4 --output output/processed_video.mp4 --codec h265
```

H.265编码相比H.264编码可以显著减小文件大小，但需要额外的编码转换时间。

## 项目结构

```
.
├── config.py             # 应用配置
├── detector/             # 检测模块
│   ├── __init__.py
│   └── pose_detector.py  # 人体姿态检测器
├── models/               # 模型文件目录（不应提交到Git仓库）
│   └── yolo11s-pose.pt   # 人体姿态检测模型 (推荐)
├── mosaic/               # 打码模块
│   ├── __init__.py
│   └── feathered_blur.py # 带羽化的高斯模糊处理
├── output/               # 处理后的视频输出目录
├── processor/            # 处理器模块
│   ├── __init__.py
│   └── video_processor.py # 视频处理主流程
├── process_videos.py     # 主处理脚本
├── requirements.txt      # 依赖包列表
├── README.md             # 说明文档
├── run.sh                # 启动脚本
├── utils/                # 工具模块
│   ├── __init__.py
│   └── video_utils.py    # 视频处理工具
└── videos/               # 示例视频文件目录
```

## 性能优化

- 在生产环境中使用NVIDIA GPU可显著提高处理速度
- 系统会自动利用多核CPU进行并行处理
- 对于大尺寸视频，可以考虑先缩小分辨率再处理
- 关键帧+插值方法可以在保持处理质量的同时减少计算量

## 注意事项

- 输出视频格式为MP4
- 处理时间取决于视频长度、分辨率和硬件性能
- 建议在处理前备份原始视频文件
- 关键帧间隔参数控制检测频率，较小的值会增加计算量但提高准确性
- 处理器输出实际打码的帧数，在调试日志中可以查看
- 在调试模式下，检测框会绘制在模糊处理之后，确保清晰可见
