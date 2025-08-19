"""
可视化工具模块
用于绘制检测结果和生成可视化输出
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import Counter


def draw_emotion_results(image, results, show_confidence=True, show_emotion=True):
	"""
	在图片上绘制情绪检测结果
	
	Args:
		image: 输入图片 (numpy.ndarray, BGR)
		results: 检测结果列表 [{'bbox':[x1,y1,x2,y2], 'confidence':float, 'emotion':'happy', 'emotion_cn':'开心'}]
		show_confidence: 是否显示置信度
		show_emotion: 是否显示情绪标签
		
	Returns:
		numpy.ndarray: 标注后的图片 (BGR)
	"""
	annotated_image = image.copy()
	
	for result in results:
		bbox = result.get('bbox', [0, 0, 0, 0])
		confidence = float(result.get('confidence', 0.0))
		emotion_en = result.get('emotion', 'unknown')
		emotion_cn = result.get('emotion_cn', emotion_en)
		
		x1, y1, x2, y2 = map(int, bbox)
		
		# 绘制边界框
		color = get_emotion_color(emotion_en)
		cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
		
		# 准备标签文本
		label_parts = []
		if show_emotion:
			label_parts.append(str(emotion_cn))
		if show_confidence:
			label_parts.append(f"{confidence:.2f}")
		label_text = " | ".join(label_parts) if label_parts else ""
		
		if label_text:
			(label_width, label_height), baseline = cv2.getTextSize(
				label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
			)
			# 标签背景
			top = max(0, y1 - label_height - 8)
			cv2.rectangle(
				annotated_image,
				(x1, top),
				(x1 + label_width + 6, y1),
				color,
				-1,
			)
			# 文本
			cv2.putText(
				annotated_image,
				label_text,
				(x1 + 3, y1 - 4),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(255, 255, 255),
				2,
			)
	
	return annotated_image


def get_emotion_color(emotion):
	"""
	根据情绪类型返回对应的颜色 (BGR)
	"""
	color_map = {
		'happy': (0, 255, 0),       # 绿色 - 开心
		'sad': (255, 0, 0),         # 蓝色 - 难过
		'angry': (0, 0, 255),       # 红色 - 愤怒
		'fear': (128, 0, 128),      # 紫色 - 恐惧
		'surprise': (0, 255, 255),  # 黄色 - 惊讶
		'disgust': (0, 128, 128),   # 深青色 - 厌恶
		'neutral': (128, 128, 128), # 灰色 - 平静
		'contempt': (0, 0, 128),    # 深蓝色 - 轻蔑
		'confused': (128, 128, 0),  # 橄榄色 - 困惑
		'helpless': (64, 64, 64),   # 深灰色 - 无助
		'bitter': (0, 64, 64),      # 深绿色 - 苦涩
		'anxious': (128, 0, 0),     # 深红色 - 焦虑
		'excited': (0, 128, 0),     # 深绿色 - 兴奋
		'calm': (128, 255, 128),    # 浅绿色 - 镇静
		'worried': (64, 0, 0),      # 深红色 - 担心
	}
	return color_map.get(emotion, (128, 128, 128))


def create_emotion_chart(results, save_path=None):
	"""
	创建情绪分布图表 (饼图 + 柱状图)
	
	Args:
		results: 检测结果列表
		save_path: 保存路径 (可选)
	"""
	# 统计情绪数量（使用英文key，显示中文）
	emotions_en = [r.get('emotion', 'unknown') for r in results]
	counts = Counter(emotions_en)
	if not counts:
		print("没有检测到情绪数据")
		return
	
	emotions = list(counts.keys())
	values = list(counts.values())
	labels_cn = [results[0].get('emotion_labels_cn', {}).get(e, None) for e in emotions] if results else None
	# 如果没有映射，直接英文
	labels = labels_cn if labels_cn and any(labels_cn) else emotions
	
	plt.figure(figsize=(16, 7))
	
	# 饼图
	plt.subplot(1, 2, 1)
	plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
	plt.title('情绪分布（饼图）')
	
	# 柱状图
	plt.subplot(1, 2, 2)
	bars = plt.bar(labels, values, color='skyblue')
	plt.title('情绪数量统计（柱状图）')
	plt.xlabel('情绪类型')
	plt.ylabel('数量')
	plt.xticks(rotation=30)
	for bar, v in zip(bars, values):
		h = bar.get_height()
		plt.text(bar.get_x() + bar.get_width() / 2, h + 0.05, str(v), ha='center', va='bottom')
	
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"图表已保存到: {save_path}")
	plt.show()


def create_emotion_timeline(video_frame_results, save_path=None):
	"""
	创建视频情绪时间线图表
	
	Args:
		video_frame_results: 按帧的检测结果列表 [{'frame_index':int, 'results':[...]}]
		save_path: 保存路径 (可选)
	"""
	if not video_frame_results:
		print("没有视频检测结果")
		return
	
	timeline = []
	emotions = []
	for frame_result in video_frame_results:
		frame_idx = frame_result.get('frame_index')
		frame_dets = frame_result.get('results', [])
		if not frame_dets:
			continue
		# 简单取第一个检测的情绪，可根据需求改为投票或最高置信度
		emotions.append(frame_dets[0].get('emotion', 'unknown'))
		timeline.append(frame_idx)
	
	if not timeline:
		print("没有有效的时间线数据")
		return
	
	# 将情绪映射为数值
	uniq = list(dict.fromkeys(emotions))
	mapping = {e: i for i, e in enumerate(uniq)}
	y_vals = [mapping[e] for e in emotions]
	
	plt.figure(figsize=(14, 5))
	plt.plot(timeline, y_vals, 'o-', linewidth=2, markersize=5)
	plt.yticks(range(len(uniq)), uniq)
	plt.xlabel('帧序号')
	plt.ylabel('情绪')
	plt.title('视频情绪时间线')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"时间线图表已保存到: {save_path}")
	plt.show()


def save_detection_summary(results, output_path):
	"""
	保存检测结果摘要到文本文件
	"""
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write("人脸情绪检测结果摘要\n")
		f.write("=" * 50 + "\n\n")
		
		total = len(results)
		f.write(f"总检测数量: {total}\n\n")
		
		counts = Counter([r.get('emotion_cn', r.get('emotion', 'unknown')) for r in results])
		f.write("情绪分布:\n")
		for emo, cnt in counts.most_common():
			percent = (cnt / total * 100) if total else 0
			f.write(f"  {emo}: {cnt} ({percent:.1f}%)\n")
		
		f.write("\n详细检测结果:\n")
		f.write("-" * 30 + "\n")
		for i, r in enumerate(results, 1):
			bbox = r.get('bbox', [])
			emo = r.get('emotion_cn', r.get('emotion', 'unknown'))
			conf = float(r.get('confidence', 0.0))
			f.write(f"检测 {i}:\n")
			f.write(f"  位置: {bbox}\n")
			f.write(f"  情绪: {emo}\n")
			f.write(f"  置信度: {conf:.3f}\n\n")
	print(f"检测摘要已保存到: {output_path}")
