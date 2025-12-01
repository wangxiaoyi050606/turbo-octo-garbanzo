import cv2
import numpy as np
import os
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 设置中文显示
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 使用PIL来绘制中文文本
def put_chinese_text(img, text, position, font_size=12, color=(0, 0, 0)):
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试使用Windows系统自带的中文字体
    try:
        # 使用Windows系统字体目录中的中文字体
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体字体
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 如果找不到指定字体，使用默认字体
        font = ImageFont.load_default()

    # 绘制文本
    draw.text(position, text, font=font, fill=color)

    # 将PIL图像转换回OpenCV图像
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# 手势类别映射
# 重要：必须与训练数据的CLASS_MAPPING一致！
# A=石头(0), B=布(1), E=剪刀(2)
class_names = {
    0: 'A',  # 石头(Rock)
    1: 'B',  # 布(Paper)
    2: 'E',  # 剪刀(Scissors)
}

# 手势中文名称映射
class_names_cn = {
    0: '石头',  # A - Rock
    1: '布',    # B - Paper
    2: '剪刀',  # E - Scissors
}


def load_model(model_path=None):
    """
    加载训练好的模型
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.dirname(script_dir)

    # 如果没有指定模型路径，尝试自动查找最新训练的模型
    if model_path is None:
        # 构建绝对路径进行查找（优先使用增量训练模型）
        possible_paths = [
            os.path.join(root_dir, "models", "output", "rps_incremental_model", "best_incremental.pt"),
            os.path.join(root_dir, "models", "output", "rps_incremental_model", "incremental_train", "weights", "best.pt"),
            os.path.join(root_dir, "models", "output", "rps_final_model", "train", "weights", "best.pt"),
            os.path.join(root_dir, "models", "output", "rps_final_model", "best.pt"),
            os.path.join(script_dir, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(root_dir, "best.pt"),
            os.path.join(script_dir, "best.pt")
        ]

        # 打印查找路径信息
        print("正在查找模型文件...")
        print(f"当前脚本目录: {script_dir}")
        print(f"项目根目录: {root_dir}")
        print("查找路径列表:")
        for i, path in enumerate(possible_paths, 1):
            exists = "✓ 存在" if os.path.exists(path) else "✗ 不存在"
            print(f"{i}. {path} - {exists}")

        # 查找存在的模型文件
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"找到模型文件: {model_path}")
                break

    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        return YOLO(model_path)
    else:
        print("错误: 找不到训练好的模型文件")
        print("请确保模型已训练完成并存在于以下路径之一:")
        print("1. models/output/rps_final_model/train/weights/best.pt")
        print("2. models/output/rps_final_model/best.pt")
        print("3. scripts/runs/detect/train/weights/best.pt")
        print("4. 当前目录下的best.pt")
        return None


def process_frame(frame, model, confidence_threshold=0.15):
    """
    处理单帧图像并进行手势识别
    """
    # 使用模型进行预测
    results = model(frame, stream=True, conf=confidence_threshold)

    detected_classes = []
    all_predictions = []  # 保存所有预测结果用于调试

    # 处理预测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 获取置信度
            confidence = box.conf[0].item()

            # 获取类别
            class_id = int(box.cls[0])
            class_name = class_names.get(class_id, f'未知({class_id})')
            class_name_cn = class_names_cn.get(class_id, f'未知({class_id})')

            # 保存所有预测用于调试
            all_predictions.append((class_id, class_name, confidence))

            # 记录检测到的类别
            detected_classes.append((class_id, class_name, class_name_cn, confidence))

            # 根据类别设置不同颜色的边界框
            if class_id == 0:  # 石头
                color = (0, 0, 255)  # 红色
            elif class_id == 1:  # 布
                color = (0, 255, 0)  # 绿色
            elif class_id == 2:  # 剪刀
                color = (255, 0, 0)  # 蓝色
            else:
                color = (255, 255, 0)  # 黄色

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f'{class_name_cn} ({class_name}): {confidence:.2f}'

            # 使用PIL绘制中文标签
            # 先计算文本大小以创建背景框
            # 由于PIL的textbbox在不同版本可能有差异，我们使用近似值
            text_height = 20
            text_width = len(label) * 10

            # 绘制标签背景
            cv2.rectangle(frame,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color,
                          -1)

            # 使用自定义函数绘制中文文本
            frame = put_chinese_text(frame, label, (x1, y1 - text_height - 5), font_size=14, color=(0, 0, 0))

    # 添加调试信息：显示所有预测结果（包括低置信度的）
    if all_predictions:
        debug_text = "调试信息: "
        for class_id, class_name, conf in sorted(all_predictions, key=lambda x: x[2], reverse=True):
            debug_text += f"{class_name}:{conf:.3f} "

        # 绘制调试信息
        cv2.rectangle(frame, (10, frame.shape[0] - 30),
                      (len(debug_text) * 8 + 20, frame.shape[0]), (0, 0, 0), -1)
        frame = put_chinese_text(frame, debug_text, (15, frame.shape[0] - 25), font_size=12, color=(255, 255, 255))

    return frame, detected_classes


def main():
    print("===== 实时手势识别程序 =====")
    print("使用rps-final模型进行手势识别")
    print("类别映射: A=石头(Rock), B=布(Paper), E=剪刀(Scissors)")
    print("按 'q' 键退出程序")
    print("按 't' 键降低置信度阈值")
    print("按 'y' 键提高置信度阈值")

    # 加载模型
    model = load_model()
    if model is None:
        return

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    print("摄像头已打开，开始识别...")
    # 降低置信度阈值以提高石头手势的识别率
    confidence_threshold = 0.15
    print(f"当前置信度阈值: {confidence_threshold}")

    # 帧率计算
    prev_time = 0
    fps = 0

    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头数据")
                break

            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)

            # 处理帧
            processed_frame, detected_classes = process_frame(frame, model, confidence_threshold)

            # 计算帧率
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # 显示帧率 - 英文不需要特殊处理
            cv2.putText(processed_frame,
                        f'FPS: {fps:.1f}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

            # 显示检测到的手势统计
            if detected_classes:
                gesture_summary = "检测到手势: "
                unique_classes = {}
                for class_id, class_name, class_name_cn, conf in detected_classes:
                    if class_id not in unique_classes or conf > unique_classes[class_id][3]:
                        unique_classes[class_id] = (class_id, class_name, class_name_cn, conf)

                gesture_names = [f"{info[2]}({info[1]})" for info in unique_classes.values()]
                gesture_summary += ", ".join(gesture_names)

                # 显示手势统计
                # 使用自定义函数绘制中文文本
                processed_frame = put_chinese_text(processed_frame, gesture_summary, (10, 50), font_size=14,
                                                   color=(255, 0, 0))
            else:
                # 当未检测到任何手势时显示提示
                processed_frame = put_chinese_text(processed_frame, "未检测到手势，请调整姿势或降低置信度阈值",
                                                   (10, 50), font_size=14, color=(0, 0, 255))

            # 显示图像 - 使用英文窗口标题以避免中文显示问题
            cv2.imshow('RPS Gesture Recognition', processed_frame)

            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出程序")
                break
            elif key == ord('t'):
                # 降低置信度阈值
                confidence_threshold = max(0.05, confidence_threshold - 0.05)
                print(f"置信度阈值已降低: {confidence_threshold}")
            elif key == ord('y'):
                # 提高置信度阈值
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"置信度阈值已提高: {confidence_threshold}")

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已关闭")


if __name__ == "__main__":
    main()