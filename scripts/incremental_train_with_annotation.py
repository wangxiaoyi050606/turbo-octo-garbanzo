"""
增量训练与实时数据标注工具
功能:
1. 基于已有模型权重进行增量训练
2. 打开摄像头进行实时数据采集和标注
3. 自动保存标注数据到YOLO格式
"""

import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import shutil
from PIL import Image, ImageDraw, ImageFont

# 类别映射（与convert_cco_to_yolo.py保持一致）
CLASS_MAPPING = {
    'A': 0,  # 石头(Rock)
    'B': 1,  # 布(Paper)
    'E': 2,  # 剪刀(Scissors)
}

CLASS_NAMES = ['A', 'B', 'E']
CLASS_DESCRIPTIONS = {
    'A': '石头(Rock)',
    'B': '布(Paper)',
    'E': '剪刀(Scissors)'
}

class DataAnnotationTool:
    """实时数据标注工具"""
    
    def __init__(self, output_dir='../data/yolo_rps_final/images/incremental'):
        """
        初始化标注工具
        
        Args:
            output_dir: 标注数据输出目录
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(script_dir)
        self.output_dir = os.path.join(self.project_root, 'data', 'yolo_rps_final', 'images', 'incremental')
        self.labels_dir = os.path.join(self.project_root, 'data', 'yolo_rps_final', 'labels', 'incremental')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        self.current_class = 0  # 当前选择的类别
        self.drawing = False  # 是否正在绘制边界框
        self.start_point = None
        self.end_point = None
        self.boxes = []  # 存储当前帧的所有标注框 [(class_id, x1, y1, x2, y2), ...]
        self.current_frame = None
        self.temp_frame = None
        self.saved_count = 0
        
        print(f'标注数据将保存到: {self.output_dir}')
        print(f'标签数据将保存到: {self.labels_dir}')
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于绘制边界框"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # 添加边界框到列表
            if self.start_point is not None and self.end_point is not None:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 确保边界框有一定大小
                    self.boxes.append((self.current_class, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                    print(f'添加标注框: 类别={CLASS_NAMES[self.current_class]} ({CLASS_DESCRIPTIONS[CLASS_NAMES[self.current_class]]})')
            
            self.start_point = None
            self.end_point = None
    
    def put_chinese_text(self, img, text, position, font_size=20, color=(0, 255, 255)):
        """在图像上绘制中文文字"""
        # 将OpenCV图像转为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 使用系统字体（Windows默认字体）
        try:
            font = ImageFont.truetype("msyh.ttc", font_size)  # 微软雅黑
        except:
            try:
                font = ImageFont.truetype("simhei.ttf", font_size)  # 黑体
            except:
                font = ImageFont.load_default()
        
        # 绘制文字
        draw.text(position, text, font=font, fill=color[::-1])  # BGR转RGB
        
        # 转回OpenCV图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def draw_boxes(self, frame):
        """在帧上绘制所有边界框"""
        display_frame = frame.copy()
        
        # 绘制已保存的边界框
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # A=绿色, B=蓝色, E=红色
        for class_id, x1, y1, x2, y2 in self.boxes:
            color = colors[class_id]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f'{CLASS_NAMES[class_id]}: {CLASS_DESCRIPTIONS[CLASS_NAMES[class_id]]}'
            # 使用中文绘制函数
            display_frame = self.put_chinese_text(display_frame, label, (x1, max(10, y1 - 25)), 18, color)
        
        # 绘制正在绘制的边界框
        if self.drawing and self.start_point and self.end_point:
            color = colors[self.current_class]
            cv2.rectangle(display_frame, self.start_point, self.end_point, color, 2)
        
        return display_frame
    
    def save_annotation(self, frame, boxes):
        """保存标注的图像和标签"""
        if len(boxes) == 0:
            print('警告: 没有标注框，跳过保存')
            return False
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_name = f'annotated_{timestamp}.jpg'
        label_name = f'annotated_{timestamp}.txt'
        
        image_path = os.path.join(self.output_dir, image_name)
        label_path = os.path.join(self.labels_dir, label_name)
        
        # 保存图像
        cv2.imwrite(image_path, frame)
        
        # 保存YOLO格式标签
        h, w = frame.shape[:2]
        with open(label_path, 'w', encoding='utf-8') as f:
            for class_id, x1, y1, x2, y2 in boxes:
                # 转换为YOLO格式（归一化中心坐标和宽高）
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n')
        
        self.saved_count += 1
        print(f'✓ 保存成功 [{self.saved_count}]: {image_name} ({len(boxes)} 个标注框)')
        return True
    
    def show_help(self, frame):
        """在帧上显示帮助信息"""
        help_text = [
            "=== 数据标注工具 ===",
            f"当前类别: {CLASS_NAMES[self.current_class]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[self.current_class]]}",
            "操作说明:",
            "  鼠标左键拖拽: 绘制边界框",
            "  A/B/E: 切换类别 (石头/布/剪刀)",
            "  S: 保存当前帧和标注",
            "  C: 清除当前帧所有标注",
            "  Q/ESC: 退出",
            f"已保存: {self.saved_count} 张图像"
        ]
        
        y_offset = 25
        for i, text in enumerate(help_text):
            frame = self.put_chinese_text(frame, text, (10, y_offset + i * 25), 18, (0, 255, 255))
        
        return frame
    
    def run(self, camera_id=0):
        """运行标注工具"""
        print('\n启动摄像头标注工具...')
        print(f'摄像头ID: {camera_id}')
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f'错误: 无法打开摄像头 {camera_id}')
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        window_name = 'RPS Data Annotation Tool'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print('\n标注工具已启动！')
        print('=' * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print('错误: 无法读取摄像头画面')
                break
            
            self.current_frame = frame.copy()
            
            # 绘制边界框
            display_frame = self.draw_boxes(frame)
            
            # 显示帮助信息
            display_frame = self.show_help(display_frame)
            
            cv2.imshow(window_name, display_frame)
            
            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q或ESC退出
                print('\n退出标注工具')
                break
            
            elif key == ord('a') or key == ord('A'):  # 切换到类别A(石头)
                self.current_class = 0
                print(f'切换类别: {CLASS_NAMES[self.current_class]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[self.current_class]]}')
            
            elif key == ord('b') or key == ord('B'):  # 切换到类别B(布)
                self.current_class = 1
                print(f'切换类别: {CLASS_NAMES[self.current_class]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[self.current_class]]}')
            
            elif key == ord('e') or key == ord('E'):  # 切换到类别E(剪刀)
                self.current_class = 2
                print(f'切换类别: {CLASS_NAMES[self.current_class]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[self.current_class]]}')
            
            elif key == ord('s') or key == ord('S'):  # 保存
                self.save_annotation(self.current_frame, self.boxes)
                self.boxes = []  # 清空标注框
            
            elif key == ord('c') or key == ord('C'):  # 清除
                self.boxes = []
                print('清除当前所有标注框')
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f'\n标注完成！共保存 {self.saved_count} 张图像')


def incremental_train(model_path, epochs=50, batch_size=8, device='cuda'):
    """
    增量训练函数
    
    Args:
        model_path: 已训练模型权重路径
        epochs: 训练轮数
        batch_size: 批次大小
        device: 训练设备
    """
    print('\n' + '=' * 60)
    print('开始增量训练')
    print('=' * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 检查模型权重是否存在
    if not os.path.exists(model_path):
        print(f'错误: 模型权重文件不存在: {model_path}')
        return None
    
    # 加载已训练的模型
    print(f'加载模型权重: {model_path}')
    model = YOLO(model_path)
    
    # 数据集配置文件
    data_config = os.path.join(project_root, 'data', 'yolo_rps_final', 'rps_final.yaml')
    
    if not os.path.exists(data_config):
        print(f'错误: 配置文件不存在: {data_config}')
        return None
    
    # 输出目录
    output_dir = os.path.join(project_root, 'models', 'output', 'rps_incremental_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查GPU
    if device == 'cuda' and not torch.cuda.is_available():
        print('警告: CUDA不可用，切换到CPU')
        device = 'cpu'
    
    print(f'使用设备: {device}')
    print(f'数据集配置: {data_config}')
    print(f'训练轮数: {epochs}')
    print(f'批次大小: {batch_size}')
    print(f'输出目录: {output_dir}')
    
    # 开始增量训练
    print('\n开始训练...')
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        workers=4,
        device=device,
        project=output_dir,
        name='incremental_train',
        exist_ok=True,
        patience=20,
        save_period=5,
        amp=False,
        resume=False,  # 不恢复训练，而是基于权重继续
        pretrained=True  # 使用预训练权重
    )
    
    # 保存最佳模型
    best_model_path = os.path.join(output_dir, 'incremental_train', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        link_path = os.path.join(output_dir, 'best_incremental.pt')
        if os.path.exists(link_path):
            os.remove(link_path)
        try:
            shutil.copy2(best_model_path, link_path)
            print(f'\n✓ 最佳模型已保存至: {link_path}')
        except Exception as e:
            print(f'保存模型失败: {e}')
    
    print('\n增量训练完成！')
    return results


def main():
    parser = argparse.ArgumentParser(description='增量训练与实时数据标注工具')
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 标注模式
    annotate_parser = subparsers.add_parser('annotate', help='启动数据标注工具')
    annotate_parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0)')
    
    # 训练模式
    train_parser = subparsers.add_parser('train', help='增量训练模型')
    train_parser.add_argument('--model', type=str, required=True, help='已训练模型权重路径')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    train_parser.add_argument('--batch', type=int, default=8, help='批次大小 (默认: 8)')
    train_parser.add_argument('--device', type=str, default='cuda', 
                            choices=['cuda', 'cpu'], help='训练设备 (默认: cuda)')
    
    # 组合模式：先标注后训练
    combined_parser = subparsers.add_parser('both', help='先标注数据，然后增量训练')
    combined_parser.add_argument('--model', type=str, required=True, help='已训练模型权重路径')
    combined_parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0)')
    combined_parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    combined_parser.add_argument('--batch', type=int, default=8, help='批次大小 (默认: 8)')
    combined_parser.add_argument('--device', type=str, default='cuda', 
                            choices=['cuda', 'cpu'], help='训练设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    if args.mode == 'annotate':
        # 仅标注数据
        tool = DataAnnotationTool()
        tool.run(camera_id=args.camera)
    
    elif args.mode == 'train':
        # 仅增量训练
        incremental_train(
            model_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device
        )
    
    elif args.mode == 'both':
        # 先标注后训练
        print('模式: 先标注数据，然后增量训练')
        print('=' * 60)
        
        # 1. 标注数据
        tool = DataAnnotationTool()
        tool.run(camera_id=args.camera)
        
        # 2. 询问是否继续训练
        if tool.saved_count > 0:
            print(f'\n已标注 {tool.saved_count} 张图像')
            response = input('是否开始增量训练? (y/n): ').strip().lower()
            
            if response == 'y':
                incremental_train(
                    model_path=args.model,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    device=args.device
                )
            else:
                print('取消训练')
        else:
            print('没有标注数据，跳过训练')
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
