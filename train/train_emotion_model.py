"""
训练情绪检测模型
使用自定义数据集训练YOLO模型
"""

import os
import yaml
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EmotionModelTrainer:
    def __init__(self, data_config_path, model_config_path):
        """
        初始化训练器
        
        Args:
            data_config_path (str): 数据配置文件路径
            model_config_path (str): 模型配置文件路径
        """
        self.data_config_path = data_config_path
        self.model_config_path = model_config_path
        
        # 创建训练目录
        self.train_dir = Path("runs/train")
        self.train_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset(self, dataset_path):
        """
        准备训练数据集
        
        Args:
            dataset_path (str): 数据集路径
        """
        logger.info("准备训练数据集...")
        
        # 创建YOLO格式的目录结构
        yolo_dir = Path("dataset/yolo_format")
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 复制图片和标签文件
        self._copy_dataset_files(dataset_path, yolo_dir)
        
        # 创建数据配置文件
        self._create_data_yaml(yolo_dir)
        
        logger.info("数据集准备完成")
    
    def _copy_dataset_files(self, source_path, target_path):
        """复制数据集文件"""
        source = Path(source_path)
        
        # 复制训练集
        train_images = list((source / "train" / "images").glob("*.jpg"))
        train_labels = list((source / "train" / "labels").glob("*.txt"))
        
        for img_file in train_images:
            shutil.copy2(img_file, target_path / "images" / "train")
        
        for label_file in train_labels:
            shutil.copy2(label_file, target_path / "labels" / "train")
        
        # 复制验证集
        val_images = list((source / "val" / "images").glob("*.jpg"))
        val_labels = list((source / "val" / "labels").glob("*.txt"))
        
        for img_file in val_images:
            shutil.copy2(img_file, target_path / "images" / "val")
        
        for label_file in val_labels:
            shutil.copy2(label_file, target_path / "labels" / "val")
        
        logger.info(f"复制了 {len(train_images)} 张训练图片, {len(val_images)} 张验证图片")
    
    def _create_data_yaml(self, dataset_path):
        """创建数据配置文件"""
        data_config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 15,  # 情绪类别数量
            'names': [
                'angry', 'disgust', 'fear', 'happy', 'sad',
                'surprise', 'neutral', 'contempt', 'confused',
                'helpless', 'bitter', 'anxious', 'excited',
                'calm', 'worried'
            ]
        }
        
        config_path = dataset_path / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"数据配置文件已创建: {config_path}")
    
    def train_model(self, epochs=100, batch_size=16, img_size=640):
        """
        训练模型
        
        Args:
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            img_size (int): 图片尺寸
        """
        logger.info("开始训练模型...")
        
        try:
            from ultralytics import YOLO
            
            # 加载预训练模型
            model = YOLO('yolov8n.pt')
            
            # 开始训练
            results = model.train(
                data='dataset/yolo_format/data.yaml',
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=0,
                project='runs/train',
                name='emotion_detection'
            )
            
            logger.info("模型训练完成")
            return results
            
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise
    
    def evaluate_model(self, model_path):
        """
        评估模型性能
        
        Args:
            model_path (str): 模型文件路径
        """
        logger.info("开始评估模型...")
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # 在验证集上评估
            results = model.val(data='dataset/yolo_format/data.yaml')
            
            logger.info("模型评估完成")
            return results
            
        except Exception as e:
            logger.error(f"模型评估过程中出错: {e}")
            raise
    
    def export_model(self, model_path, export_format='onnx'):
        """
        导出模型
        
        Args:
            model_path (str): 模型文件路径
            export_format (str): 导出格式
        """
        logger.info(f"导出模型为 {export_format} 格式...")
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # 导出模型
            exported_model = model.export(format=export_format)
            
            logger.info(f"模型导出完成: {exported_model}")
            return exported_model
            
        except Exception as e:
            logger.error(f"模型导出过程中出错: {e}")
            raise

def main():
    """主训练函数"""
    # 配置路径
    data_config = "config/dataset.yaml"
    model_config = "config/model.yaml"
    
    # 创建训练器
    trainer = EmotionModelTrainer(data_config, model_config)
    
    # 准备数据集
    dataset_path = "dataset/emotion_dataset"
    if os.path.exists(dataset_path):
        trainer.prepare_dataset(dataset_path)
    
    # 训练模型
    results = trainer.train_model(epochs=100, batch_size=16)
    
    # 评估模型
    best_model = "runs/train/emotion_detection/weights/best.pt"
    if os.path.exists(best_model):
        trainer.evaluate_model(best_model)
        
        # 导出模型
        trainer.export_model(best_model, 'onnx')

if __name__ == "__main__":
    main()
