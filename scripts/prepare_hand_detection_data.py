import os
import shutil
import cv2
import numpy as np
import random
from tqdm import tqdm

# 检查是否可以使用CUDA加速OpenCV
try:
    # 检查OpenCV是否有CUDA支持
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"OpenCV CUDA支持: {'可用' if cuda_available else '不可用'}")
    if cuda_available:
        print(f"CUDA设备数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
except:
    cuda_available = False
    print("OpenCV CUDA模块不可用，将使用CPU处理")

# 设置路径
RAW_DATASET_PATH = r"../dataset/Rock-Paper-Scissors"
YOLO_DATASET_PATH = r"../data/yolo_hand_detection"
TRAIN_IMAGES_PATH = os.path.join(YOLO_DATASET_PATH, "images", "train")
VAL_IMAGES_PATH = os.path.join(YOLO_DATASET_PATH, "images", "val")
TRAIN_LABELS_PATH = os.path.join(YOLO_DATASET_PATH, "labels", "train")
VAL_LABELS_PATH = os.path.join(YOLO_DATASET_PATH, "labels", "val")

# 创建目录
os.makedirs(TRAIN_IMAGES_PATH, exist_ok=True)
os.makedirs(VAL_IMAGES_PATH, exist_ok=True)
os.makedirs(TRAIN_LABELS_PATH, exist_ok=True)
os.makedirs(VAL_LABELS_PATH, exist_ok=True)

# 手势类别
GESTURE_CLASSES = ["rock", "paper", "scissors"]

# 简单的手部检测方法（基于肤色和轮廓）
def detect_hand_bbox(image):
    # 如果OpenCV有CUDA支持，使用CUDA加速处理
    if 'cuda_available' in globals() and cuda_available:
        try:
            # 将图像上传到GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # 转换到HSV色彩空间
            gpu_hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)
            
            # 肤色范围（简化版）
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # 创建肤色掩码
            gpu_mask = cv2.cuda.inRange(gpu_hsv, lower_skin, upper_skin)
            
            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            gpu_kernel = cv2.cuda_GpuMat()
            gpu_kernel.upload(kernel)
            
            # 闭操作和开操作
            morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, gpu_kernel)
            gpu_mask = morph.apply(gpu_mask)
            
            morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, gpu_kernel)
            gpu_mask = morph.apply(gpu_mask)
            
            # 下载回CPU进行轮廓查找（OpenCV CUDA模块没有直接的轮廓查找功能）
            mask = gpu_mask.download()
        except Exception as e:
            print(f"CUDA处理出错，回退到CPU: {e}")
            # 如果CUDA处理失败，回退到CPU处理
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 肤色范围（简化版）
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # 创建肤色掩码
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        # CPU处理
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 肤色范围（简化版）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建肤色掩码
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓（目前只能在CPU上进行）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的轮廓（假设是手）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 确保边界框有效且足够大
        if w > 50 and h > 50:
            return x, y, w, h
    
    # 如果没有检测到手，返回图像中心的一个默认区域
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 2
    return center_x - size // 2, center_y - size // 2, size, size

# 将边界框转换为YOLO格式
def bbox_to_yolo_format(x, y, w, h, img_width, img_height):
    # YOLO格式：class_idx center_x center_y width height
    # 所有值都归一化到[0, 1]
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return [0, center_x, center_y, width, height]  # 0 是hand类的索引

# 处理图像和生成标签
def process_images():
    all_images = []
    
    # 收集所有图像路径
    for gesture in GESTURE_CLASSES:
        # 检查不同的目录结构
        for base_dir in [os.path.join(RAW_DATASET_PATH, "train"), 
                         os.path.join(RAW_DATASET_PATH, "Rock-Paper-Scissors", "train")]:
            gesture_dir = os.path.join(base_dir, gesture)
            if os.path.exists(gesture_dir):
                for img_file in os.listdir(gesture_dir):
                    if img_file.endswith((".png", ".jpg", ".jpeg")):
                        all_images.append(os.path.join(gesture_dir, img_file))
        
        # 也包含测试集图像以增加数据量
        for base_dir in [os.path.join(RAW_DATASET_PATH, "test"), 
                         os.path.join(RAW_DATASET_PATH, "Rock-Paper-Scissors", "test")]:
            gesture_dir = os.path.join(base_dir, gesture)
            if os.path.exists(gesture_dir):
                for img_file in os.listdir(gesture_dir):
                    if img_file.endswith((".png", ".jpg", ".jpeg")):
                        all_images.append(os.path.join(gesture_dir, img_file))
    
    # 添加验证集图像
    val_dir = os.path.join(RAW_DATASET_PATH, "validation")
    if os.path.exists(val_dir):
        for img_file in os.listdir(val_dir):
            if img_file.endswith((".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(val_dir, img_file))
    
    # 去重
    all_images = list(set(all_images))
    print(f"总共找到 {len(all_images)} 张图像")
    
    # 划分训练集和验证集（80%训练，20%验证）
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(val_images)} 张图像")
    
    # 处理训练集
    for img_path in tqdm(train_images, desc="处理训练集"):
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # 检测手部边界框
            x, y, w, h = detect_hand_bbox(image)
            
            # 确保边界框在图像范围内
            h_img, w_img = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w_img - x, w)
            h = min(h_img - y, h)
            
            # 生成唯一的文件名
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # 复制图像到目标位置
            target_img_path = os.path.join(TRAIN_IMAGES_PATH, img_name)
            cv2.imwrite(target_img_path, image)
            
            # 生成YOLO格式的标签
            yolo_bbox = bbox_to_yolo_format(x, y, w, h, w_img, h_img)
            label_path = os.path.join(TRAIN_LABELS_PATH, f"{base_name}.txt")
            
            with open(label_path, "w") as f:
                f.write(" ".join(map(str, yolo_bbox)))
        
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    # 处理验证集
    for img_path in tqdm(val_images, desc="处理验证集"):
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # 检测手部边界框
            x, y, w, h = detect_hand_bbox(image)
            
            # 确保边界框在图像范围内
            h_img, w_img = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w_img - x, w)
            h = min(h_img - y, h)
            
            # 生成唯一的文件名
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # 复制图像到目标位置
            target_img_path = os.path.join(VAL_IMAGES_PATH, img_name)
            cv2.imwrite(target_img_path, image)
            
            # 生成YOLO格式的标签
            yolo_bbox = bbox_to_yolo_format(x, y, w, h, w_img, h_img)
            label_path = os.path.join(VAL_LABELS_PATH, f"{base_name}.txt")
            
            with open(label_path, "w") as f:
                f.write(" ".join(map(str, yolo_bbox)))
        
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    print("数据准备完成！")

if __name__ == "__main__":
    process_images()