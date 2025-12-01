import os
import cv2
import numpy as np
import torch

# 设置环境变量，防止自动下载模型
os.environ['YOLO_VERBOSE'] = '0'
os.environ['ULTRALYTICS_HUB_OFFLINE'] = '1'

def optimize_bbox_for_hand(x1, y1, x2, y2):
    """
    优化边界框，专注于手部而非手臂区域
    使用基于手部几何特征的算法来精确定位手部
    """
    # 计算原始边界框的中心点和尺寸
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    
    # 手部几何分析：
    # 1. 手部通常具有更紧凑的结构，而手臂则是延伸的
    # 2. 典型的手部边界框应该更加方形，而不是狭长的矩形
    aspect_ratio = width / height
    
    # 基于长宽比的优化策略
    if aspect_ratio < 0.6:  # 较窄的边界框，可能是手臂伸展状态
        # 假设手部位于靠近末端的位置（通常是底部）
        # 将边界框重点放在检测框的下半部分
        hand_section = 0.4  # 手部占整个检测框的比例
        new_height = int(height * hand_section)
        new_width = int(width * 0.8)  # 适当收缩宽度
        
        # 从底部向上定位手部
        new_y2 = y2
        new_y1 = max(0, new_y2 - new_height)
        new_x1 = max(0, center_x - new_width // 2)
        new_x2 = new_x1 + new_width
    
    elif aspect_ratio > 1.5:  # 较宽的边界框，可能是手掌展开
        # 手掌通常更加方形，所以调整为更接近方形的比例
        target_height = int(width * 1.2)  # 稍微高一些
        target_height = min(target_height, height)  # 不超过原始高度
        
        # 确保边界框集中在手掌区域
        new_width = int(width * 0.7)  # 收缩宽度
        new_height = target_height
        
        # 从中心定位
        new_x1 = max(0, center_x - new_width // 2)
        new_x2 = new_x1 + new_width
        new_y1 = max(0, center_y - new_height // 2)
        new_y2 = new_y1 + new_height
    
    else:  # 中等长宽比，接近方形
        # 针对更可能是手部的情况，使用更小的收缩比例
        # 手部通常在检测框的中下部
        hand_center_factor = 0.65  # 将中心点向下移动
        hand_center_y = y1 + int(height * hand_center_factor)
        
        # 收缩尺寸
        new_width = int(width * 0.65)
        new_height = int(height * 0.65)
        
        # 重新定位边界框
        new_x1 = max(0, center_x - new_width // 2)
        new_x2 = new_x1 + new_width
        new_y1 = max(0, hand_center_y - new_height // 2)
        new_y2 = new_y1 + new_height
    
    # 确保边界框不会过度缩小
    min_size = 50  # 最小尺寸阈值
    if (new_x2 - new_x1) < min_size:
        expand = (min_size - (new_x2 - new_x1)) // 2
        new_x1 = max(0, new_x1 - expand)
        new_x2 = new_x1 + min_size
    
    if (new_y2 - new_y1) < min_size:
        expand = (min_size - (new_y2 - new_y1)) // 2
        new_y1 = max(0, new_y1 - expand)
        new_y2 = new_y1 + min_size
    
    return new_x1, new_y1, new_x2, new_y2

def detect_hand_in_image(model, image_path, output_suffix):
    """
    使用模型检测单张图片中的手
    """
    print(f"\n处理图片: {image_path}")
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None, False
    
    print(f"✅ 成功读取图片")
    
    # 直接调用模型的预测方法
    print(f"🔄 正在进行模型推理...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.predict(source=image, conf=0.5, device=device, verbose=False)
    
    # 处理结果
    result_image = image.copy()
    detected = False
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        detected = True
        # 获取置信度最高的检测框
        boxes = results[0].boxes
        max_conf_idx = np.argmax(boxes.conf.cpu().numpy())
        
        # 提取原始边界框信息
        orig_x1, orig_y1, orig_x2, orig_y2 = map(int, boxes.xyxy[max_conf_idx].cpu().numpy())
        conf = float(boxes.conf[max_conf_idx].cpu().numpy())
        
        # 优化边界框，聚焦于手部
        opt_x1, opt_y1, opt_x2, opt_y2 = optimize_bbox_for_hand(orig_x1, orig_y1, orig_x2, orig_y2)
        
        # 绘制原始边界框（蓝色虚线）
        cv2.rectangle(result_image, (orig_x1, orig_y1), (orig_x2, orig_y2), (255, 0, 0), 2, cv2.LINE_AA)
        
        # 绘制优化后的边界框（绿色实线）
        cv2.rectangle(result_image, (opt_x1, opt_y1), (opt_x2, opt_y2), (0, 255, 0), 2, cv2.LINE_AA)
        
        # 添加标签
        label = f"Hand: {conf:.2f}"
        cv2.putText(result_image, label, (opt_x1, max(0, opt_y1-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        print(f"✅ 检测到手！")
        print(f"  - 原始位置: ({orig_x1}, {orig_y1}) 到 ({orig_x2}, {orig_y2})")
        print(f"  - 优化后位置: ({opt_x1}, {opt_y1}) 到 ({opt_x2}, {opt_y2})")
        print(f"  - 置信度: {conf:.2f}")
    else:
        print("❌ 未检测到手")
    
    # 保存结果
    output_path = os.path.join("..", f"simple_hand_detection_result_{output_suffix}.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"✅ 结果已保存到: {output_path}")
    
    return result_image, detected

def main():
    """
    简化版手部检测脚本 - 直接使用训练好的模型进行识别
    现在同时检测test1.jpg和test2.jpeg
    """
    print("====== 手部检测 (直接使用模型) ======\n")
    
    # 1. 加载YOLO模型
    print("正在加载训练好的手部检测模型...")
    try:
        from ultralytics import YOLO
        
        # 使用项目中训练好的模型
        model_path = os.path.join("..", "models", "output", "hand_detection_model", "weights", "best.pt")
        
        if os.path.exists(model_path):
            print(f"✅ 找到模型: {model_path}")
            model = YOLO(model_path)  # 直接加载模型
            print("✅ 模型加载成功！")
        else:
            print(f"❌ 未找到训练好的模型: {model_path}")
            print("使用默认YOLOv8n模型...")
            model = YOLO("yolov8n.pt")
            print("✅ 默认模型加载成功！")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 准备检测的图片列表 - 包括test1和test2
    print("\n准备检测图片...")
    image_paths = [
        (os.path.join("..", "test1.jpg"), "test1"),
        (os.path.join("..", "test2.jpeg"), "test2")
    ]
    
    result_images = []
    
    # 3. 对每张图片进行检测
    for image_path, suffix in image_paths:
        if os.path.exists(image_path):
            result_img, detected = detect_hand_in_image(model, image_path, suffix)
            if result_img is not None:
                result_images.append((result_img, suffix))
        else:
            print(f"❌ 图片不存在: {image_path}")
    
    # 4. 显示所有结果
    if result_images:
        print("\n显示检测结果 (按任意键关闭每个窗口)...")
        for img, suffix in result_images:
            cv2.imshow(f"手部检测结果 - {suffix}", img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n====== 所有图片检测完成 ======")
    print(f"总共处理图片数: {len(result_images)}")
    print(f"结果文件保存在项目根目录")

if __name__ == "__main__":
    main()