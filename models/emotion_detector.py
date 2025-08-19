"""
情绪检测器类
基于YOLO模型进行人脸检测和情绪分类
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path='best.pt', conf_threshold=0.5):
        """
        初始化情绪检测器
        
        Args:
            model_path (str): YOLO模型文件路径
            conf_threshold (float): 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 情绪类别
        self.emotion_classes = [
            'angry',      # 愤怒
            'disgust',    # 厌恶
            'fear',       # 恐惧
            'happy',      # 开心
            'sad',        # 难过
            'surprise',   # 惊讶
            'neutral',    # 平静
            'contempt',   # 轻蔑
            'confused',   # 困惑
            'helpless',   # 无助
            'bitter',     # 苦涩
            'anxious',    # 焦虑
            'excited',    # 兴奋
            'calm',       # 镇静
            'worried'     # 担心
        ]
        
        # 中文标签映射
        self.emotion_labels_cn = {
            'angry': '愤怒',
            'disgust': '厌恶', 
            'fear': '恐惧',
            'happy': '开心',
            'sad': '难过',
            'surprise': '惊讶',
            'neutral': '平静',
            'contempt': '轻蔑',
            'confused': '困惑',
            'helpless': '无助',
            'bitter': '苦涩',
            'anxious': '焦虑',
            'excited': '兴奋',
            'calm': '镇静',
            'worried': '担心'
        }
        
        try:
            # 加载YOLO模型
            self.model = YOLO(model_path)
            logger.info(f"模型加载成功: {model_path}")
            logger.info(f"使用设备: {self.device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def detect(self, image):
        """
        检测图片中的人脸情绪
        
        Args:
            image: 输入图片 (numpy array)
            
        Returns:
            list: 检测结果列表
        """
        try:
            # 使用YOLO进行检测
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            # 处理检测结果
            processed_results = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取置信度
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 获取类别
                        class_id = int(box.cls[0].cpu().numpy())
                        emotion = self.emotion_classes[class_id] if class_id < len(self.emotion_classes) else 'unknown'
                        
                        # 获取人脸区域
                        face_region = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # 构建检测结果
                        detection_result = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'emotion': emotion,
                            'emotion_cn': self.emotion_labels_cn.get(emotion, emotion),
                            'face_region': face_region
                        }
                        
                        processed_results.append(detection_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"检测过程中出错: {e}")
            return []
    
    def detect_batch(self, images):
        """
        批量检测图片
        
        Args:
            images (list): 图片列表
            
        Returns:
            list: 检测结果列表
        """
        batch_results = []
        
        for i, image in enumerate(images):
            logger.info(f"处理第 {i+1}/{len(images)} 张图片")
            results = self.detect(image)
            batch_results.append({
                'image_index': i,
                'results': results
            })
        
        return batch_results
    
    def get_emotion_statistics(self, results):
        """
        获取情绪统计信息
        
        Args:
            results (list): 检测结果列表
            
        Returns:
            dict: 情绪统计字典
        """
        emotion_counts = {}
        
        for result in results:
            emotion = result['cdemotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return emotion_counts
    
    def preprocess_image(self, image):
        """
        预处理图片
        
        Args:
            image: 输入图片
            
        Returns:
            numpy array: 预处理后的图片
        """
        # 调整图片大小
        if image.shape[0] > 1024 or image.shape[1] > 1024:
            scale = min(1024 / image.shape[0], 1024 / image.shape[1])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
