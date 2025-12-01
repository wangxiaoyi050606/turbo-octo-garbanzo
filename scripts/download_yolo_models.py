import os
import requests
import shutil
import hashlib
from tqdm import tqdm

# 模型下载配置
# 只包含yolov8n.pt模型，避免尝试下载链接有问题的yolo11n.pt模型
MODELS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
}

# 文件校验和（用于验证下载完整性）
MD5_HASHES = {
    "yolov8n.pt": "95a2449609c73cd69a072b09daaff0cc"  # 验证过的MD5哈希值
}

def set_mirror_source():
    """设置必要的环境变量以阻止不必要的下载和更新"""
    # 设置Ultralytics离线模式
    os.environ["ULTRALYTICS_HUB_OFFLINE"] = "1"
    os.environ["YOLO_NO_UPDATE_CHECK"] = "1"
    os.environ["ULTRALYTICS_NO_AUTOINSTALL"] = "1"
    # 设置PyTorch缓存路径
    os.environ["TORCH_HOME"] = os.path.join(os.path.expanduser("~"), ".cache/torch")

def calculate_md5(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_model(model_name, output_dir):
    """下载YOLO模型，带有进度条和完整性验证"""
    if model_name not in MODELS:
        print(f"错误：不支持的模型名称 {model_name}")
        return False
    
    url = MODELS[model_name]
    output_path = os.path.join(output_dir, model_name)
    
    # 如果文件已存在，先删除
    if os.path.exists(output_path):
        print(f"删除已存在的文件: {output_path}")
        os.remove(output_path)
    
    print(f"开始下载 {model_name} 从 {url}")
    
    try:
        # 创建会话并设置超时
        session = requests.Session()
        session.timeout = 300  # 5分钟超时
        
        # 发送请求
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            
            # 获取文件大小
            total_size = int(r.headers.get('content-length', 0))
            
            # 创建进度条
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                # 写入文件
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"下载完成: {output_path}")
        
        # 验证文件大小
        if total_size != 0 and os.path.getsize(output_path) != total_size:
            print(f"警告：文件大小不匹配，可能下载不完整")
            return False
        
        # 计算并显示MD5哈希值
        md5 = calculate_md5(output_path)
        print(f"文件 {model_name} 的MD5哈希值: {md5}")
        
        # 如果有预设的哈希值，则进行验证
        if model_name in MD5_HASHES and MD5_HASHES[model_name]:
            if md5 == MD5_HASHES[model_name]:
                print("文件完整性验证通过")
            else:
                print("警告：文件完整性验证失败，可能已损坏")
                return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        # 清理不完整的文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def download_all_models(output_dir):
    """下载所有配置的模型"""
    success_count = 0
    for model_name in MODELS.keys():
        if download_model(model_name, output_dir):
            success_count += 1
    
    print(f"\n下载完成。成功: {success_count}/{len(MODELS)}")
    return success_count == len(MODELS)

def main():
    """主函数"""
    # 设置镜像源
    set_mirror_source()
    
    # 获取脚本所在目录作为输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"模型将下载到: {output_dir}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载所有模型
    download_all_models(output_dir)
    
    # 打印完成信息
    print("\n模型下载脚本执行完成。")
    print("您现在可以在训练脚本中使用这些本地模型文件。")

if __name__ == "__main__":
    main()