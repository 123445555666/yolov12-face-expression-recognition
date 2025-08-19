"""
视频处理工具模块
用于处理视频文件的人脸情绪检测
"""

import cv2
import numpy as np
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, emotion_detector):
        """
        初始化视频处理器
        
        Args:
            emotion_detector: 情绪检测器实例
        """
        self.detector = emotion_detector
        
    def process_video(self, input_path, output_path, frame_interval=1):
        """
        处理视频文件
        
        Args:
            input_path (str): 输入视频路径
            output_path (str): 输出视频路径
            frame_interval (int): 处理帧间隔
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {input_path}")
                return
            
            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按间隔处理帧
                if frame_count % frame_interval == 0:
                    # 检测情绪
                    results = self.detector.detect(frame)
                    
                    # 绘制结果
                    from .visualization import draw_emotion_results
                    annotated_frame = draw_emotion_results(frame, results)
                    
                    processed_count += 1
                    logger.info(f"处理帧 {frame_count}/{total_frames} (进度: {frame_count/total_frames*100:.1f}%)")
                else:
                    # 直接复制帧
                    annotated_frame = frame
                
                # 写入输出视频
                out.write(annotated_frame)
            
            # 释放资源
            cap.release()
            out.release()
            
            logger.info(f"视频处理完成: {processed_count} 帧已处理")
            
        except Exception as e:
            logger.error(f"视频处理过程中出错: {e}")
            raise
    
    def process_video_batch(self, input_dir, output_dir, frame_interval=1):
        """
        批量处理视频文件
        
        Args:
            input_dir (str): 输入目录
            output_dir (str): 输出目录
            frame_interval (int): 处理帧间隔
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"在 {input_dir} 中没有找到视频文件")
            return
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        for video_file in video_files:
            logger.info(f"开始处理: {video_file.name}")
            
            output_file = output_path / f"processed_{video_file.name}"
            
            try:
                self.process_video(str(video_file), str(output_file), frame_interval)
                logger.info(f"处理完成: {video_file.name}")
            except Exception as e:
                logger.error(f"处理 {video_file.name} 时出错: {e}")
    
    def extract_frames(self, video_path, output_dir, frame_interval=30):
        """
        从视频中提取帧
        
        Args:
            video_path (str): 视频文件路径
            output_dir (str): 输出目录
            frame_interval (int): 提取帧间隔
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % frame_interval == 0:
                    # 保存帧
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = output_path / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    extracted_count += 1
            
            cap.release()
            logger.info(f"帧提取完成: 提取了 {extracted_count} 帧")
            
        except Exception as e:
            logger.error(f"帧提取过程中出错: {e}")
            raise
    
    def create_video_from_frames(self, frames_dir, output_path, fps=30):
        """
        从帧序列创建视频
        
        Args:
            frames_dir (str): 帧目录
            output_path (str): 输出视频路径
            fps (int): 帧率
        """
        try:
            frames_path = Path(frames_dir)
            frame_files = sorted(frames_path.glob("*.jpg"))
            
            if not frame_files:
                logger.error(f"在 {frames_dir} 中没有找到帧文件")
                return
            
            # 读取第一帧获取尺寸
            first_frame = cv2.imread(str(frame_files[0]))
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logger.info(f"开始创建视频: {len(frame_files)} 帧, {width}x{height}, {fps}fps")
            
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                out.write(frame)
            
            out.release()
            logger.info(f"视频创建完成: {output_path}")
            
        except Exception as e:
            logger.error(f"视频创建过程中出错: {e}")
            raise
