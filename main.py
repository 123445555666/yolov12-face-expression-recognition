#!/usr/bin/env python3
"""
人脸情绪检测主程序
支持图片、视频、实时摄像头检测
"""

import cv2
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.emotion_detector import EmotionDetector
from utils.visualization import draw_emotion_results
from utils.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description='人脸情绪检测系统')
    parser.add_argument('--source', type=str, default='0', 
                       help='输入源: 0(摄像头), 图片路径, 视频路径')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='模型文件路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化情绪检测器
    detector = EmotionDetector(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    # 处理输入源
    if args.source == '0':
        # 实时摄像头检测
        process_camera(detector, args.output)
    elif args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 视频文件检测
        process_video(args.source, detector, args.output)
    elif args.source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # 图片文件检测
        process_image(args.source, detector, args.output)
    else:
        print(f"不支持的输入源: {args.source}")

def process_camera(detector, output_dir):
    """实时摄像头检测"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出，按 's' 键保存当前帧")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测情绪
        results = detector.detect(frame)
        
        # 绘制结果
        annotated_frame = draw_emotion_results(frame, results)
        
        # 显示结果
        cv2.imshow('人脸情绪检测', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            output_path = os.path.join(output_dir, f"camera_capture_{len(os.listdir(output_dir))}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            print(f"已保存到: {output_path}")
    
    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path, detector, output_dir):
    """视频文件检测"""
    processor = VideoProcessor(detector)
    output_path = os.path.join(output_dir, f"processed_{Path(video_path).name}")
    
    print(f"正在处理视频: {video_path}")
    processor.process_video(video_path, output_path)
    print(f"处理完成，输出文件: {output_path}")

def process_image(image_path, detector, output_dir):
    """图片文件检测"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 检测情绪
    results = detector.detect(image)
    
    # 绘制结果
    annotated_image = draw_emotion_results(image, results)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
    cv2.imwrite(output_path, annotated_image)
    
    print(f"检测完成，结果保存到: {output_path}")
    
    # 显示结果
    cv2.imshow('人脸情绪检测结果', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
