import os
import argparse
from ultralytics import YOLO
import torch

def train_rps_final_model():
    """训练使用rps_final数据集的YOLO模型"""
    # 数据集配置文件路径 - 使用绝对路径避免路径解析问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_config = os.path.join(project_root, 'data', 'yolo_rps_final', 'rps_final.yaml')
    
    # 输出目录
    output_dir = os.path.join(project_root, 'models', 'output', 'rps_final_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查配置文件是否存在
    if not os.path.exists(data_config):
        print(f'错误: 配置文件 {data_config} 不存在')
        return
    
    # 检查GPU是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    
    # 加载预训练模型
    # 检查模型文件是否已存在且可用
    model_path = 'yolov8n.pt'
    if os.path.exists(model_path):
        try:
            # 尝试加载现有模型
            print(f'使用已存在的模型文件: {model_path}')
            model = YOLO(model_path)
        except Exception as e:
            # 如果加载失败，则使用内置下载功能
            print(f'模型文件加载失败: {e}')
            print('使用Ultralytics的内置下载功能获取最新版本')
            model = YOLO('yolov8n.pt') #自动下载

    else:
        # 如果文件不存在，则使用内置下载功能
        print('模型文件不存在，使用Ultralytics的内置下载功能获取')
        model = YOLO('yolov8n.pt')
    
    # 训练参数
    epochs = 200  # 增加训练轮数
    batch_size = 16
    imgsz = 640
    workers = 4
    
    print('开始训练...')
    print(f'数据集配置: {data_config}')
    print(f'训练轮数: {epochs}')
    print(f'批次大小: {batch_size}')
    print(f'图像大小: {imgsz}')
    
    # 开始训练
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        device=device,
        project=output_dir,
        name='train',
        exist_ok=True,
        patience=50,  # 增加早停耐心值，让训练更充分
        save_period=10,  # 每10个epoch保存一次
        amp=False  # 禁用自动混合精度训练以提高稳定性
    )
    
    # 保存最佳模型的软链接
    best_model_path = os.path.join(output_dir, 'train', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        # 创建到根目录的链接
        link_path = os.path.join(output_dir, 'best.pt')
        if os.path.exists(link_path):
            os.remove(link_path)
        try:
            os.symlink(best_model_path, link_path)
        except:
            # Windows可能不支持软链接，复制文件代替
            import shutil
            shutil.copy2(best_model_path, link_path)
        print(f'最佳模型已保存至: {link_path}')
    
    # 导出为ONNX格式以便部署
    try:
        onnx_path = os.path.join(output_dir, 'best.onnx')
        model.export(format='onnx', imgsz=imgsz, opset=12)
        print(f'模型已导出为ONNX格式: {onnx_path}')
    except Exception as e:
        print(f'导出ONNX失败: {str(e)}')
    
    print('训练完成!')
    return results

def main():
    parser = argparse.ArgumentParser(description='训练RPS Final数据集的YOLO模型')
    parser.add_argument('--model', type=str, default=None, help='自定义模型路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    
    args = parser.parse_args()
    
    print('=' * 50)
    print('RPS Final 数据集 YOLO模型训练')
    print('=' * 50)
    
    train_rps_final_model()

if __name__ == '__main__':
    main()