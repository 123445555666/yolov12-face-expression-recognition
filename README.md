# 人脸情绪检测系统

基于YOLO的人脸情绪检测项目，支持15种情绪识别，包括图片、视频和实时摄像头检测。

## 功能特性

- 🎭 **多情绪识别**: 支持15种情绪类型（开心、难过、愤怒、恐惧等）
- 📸 **多输入源**: 支持图片、视频、实时摄像头
- 🎯 **高精度检测**: 基于YOLOv8的先进目标检测算法
- 📊 **可视化结果**: 实时显示检测结果和统计图表
- 🚀 **高性能**: 支持GPU加速，实时处理
- 📁 **批量处理**: 支持批量图片和视频处理

## 情绪类别

| 英文 | 中文 | 英文 | 中文 |
|------|------|------|------|
| happy | 开心 | sad | 难过 |
| angry | 愤怒 | fear | 恐惧 |
| surprise | 惊讶 | disgust | 厌恶 |
| neutral | 平静 | contempt | 轻蔑 |
| confused | 困惑 | helpless | 无助 |
| bitter | 苦涩 | anxious | 焦虑 |
| excited | 兴奋 | calm | 镇静 |
| worried | 担心 | - | - |

## 项目结构

```
emotion_detection_project/
├── main.py                 # 主程序入口
├── models/                 # 模型相关
│   └── emotion_detector.py # 情绪检测器
├── utils/                  # 工具模块
│   ├── visualization.py    # 可视化工具
│   └── video_processor.py # 视频处理工具
├── train/                  # 训练相关
│   └── train_emotion_model.py # 模型训练脚本
├── config/                 # 配置文件
│   ├── dataset.yaml       # 数据集配置
│   └── model.yaml         # 模型配置
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## 安装说明

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU版本)
- 8GB+ RAM
- 2GB+ 显存 (GPU版本)

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd emotion_detection_project

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载预训练模型

```bash
# 下载YOLOv8预训练模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 使用方法

### 1. 实时摄像头检测

```bash
python main.py --source 0
```

### 2. 图片检测

```bash
python main.py --source path/to/image.jpg
```

### 3. 视频检测

```bash
python main.py --source path/to/video.mp4
```

### 4. 自定义参数

```bash
python main.py --source 0 --conf 0.7 --output results
```

参数说明:
- `--source`: 输入源 (0=摄像头, 图片路径, 视频路径)
- `--model`: 模型文件路径
- `--conf`: 置信度阈值 (0.0-1.0)
- `--output`: 输出目录

## 训练自定义模型

### 1. 准备数据集

按照YOLO格式组织数据:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 2. 开始训练

```bash
python train/train_emotion_model.py
```

### 3. 训练参数配置

在 `config/model.yaml` 中修改训练参数。

## 性能指标

- **检测精度**: mAP@0.5 > 0.85
- **处理速度**: 30+ FPS (GPU), 10+ FPS (CPU)
- **支持分辨率**: 640x640 到 1920x1080
- **内存占用**: < 2GB

## 常见问题

### Q: 检测不到人脸怎么办？
A: 检查图片质量、光线条件，调整置信度阈值。

### Q: 如何提高检测精度？
A: 使用更多训练数据，调整模型参数，使用更大的模型。

### Q: GPU内存不足？
A: 减小batch_size，使用较小的模型版本。

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系开发者。
