#!/bin/bash
# 启动脚本

# 激活虚拟环境
source .venv/bin/activate

# 运行处理脚本
python process_videos.py "$@"