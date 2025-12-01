import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm

# 设置路径
DATASET_ROOT = os.path.join('..', 'rps_final')
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, 'Annotations')
IMAGES_DIR = os.path.join(DATASET_ROOT, 'Images')
TRAIN_LIST_PATH = os.path.join(DATASET_ROOT, 'train1.txt')
VAL_LIST_PATH = os.path.join(DATASET_ROOT, 'val1.txt')
TEST_LIST_PATH = os.path.join(DATASET_ROOT, 'test1.txt')

# 输出路径
OUTPUT_DIR = os.path.join('..', 'data', 'yolo_rps_final')
OUTPUT_IMAGES_TRAIN = os.path.join(OUTPUT_DIR, 'images', 'train')
OUTPUT_IMAGES_VAL = os.path.join(OUTPUT_DIR, 'images', 'val')
OUTPUT_IMAGES_TEST = os.path.join(OUTPUT_DIR, 'images', 'test')
OUTPUT_LABELS_TRAIN = os.path.join(OUTPUT_DIR, 'labels', 'train')
OUTPUT_LABELS_VAL = os.path.join(OUTPUT_DIR, 'labels', 'val')
OUTPUT_LABELS_TEST = os.path.join(OUTPUT_DIR, 'labels', 'test')

# 类别映射（根据XML中的name字段）
# 重要：确保与实时检测脚本中的映射一致！
# 正确的类别定义：
# A = 石头(Rock), B = 布(Paper), E = 剪刀(Scissors)
CLASS_MAPPING = {
    'A': 0,  # 石头(Rock)
    'B': 1,  # 布(Paper)
    'E': 2,  # 剪刀(Scissors)
}

def create_directories():
    """创建输出目录结构"""
    directories = [
        OUTPUT_IMAGES_TRAIN, OUTPUT_IMAGES_VAL, OUTPUT_IMAGES_TEST,
        OUTPUT_LABELS_TRAIN, OUTPUT_LABELS_VAL, OUTPUT_LABELS_TEST
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f'创建目录: {dir_path}')

def read_list_file(list_path):
    """读取训练/验证/测试列表文件"""
    if not os.path.exists(list_path):
        print(f'警告: {list_path} 不存在')
        return []
    
    file_pairs = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # 每行包含图像路径和标注路径，用空格分隔
            parts = line.strip().split()
            if len(parts) >= 2:
                # 提取文件名（不包含路径和扩展名）
                image_path = parts[0]
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                file_pairs.append(file_name)
    return file_pairs

def convert_xml_to_yolo(xml_file, output_label_path):
    """将单个XML文件转换为YOLO格式的txt文件"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 处理每个对象
        yolo_lines = []
        for obj in root.findall('object'):
            # 获取类别
            name = obj.find('name').text
            if name not in CLASS_MAPPING:
                print(f'警告: 未知类别 {name} 在 {xml_file} 中')
                continue
            class_id = CLASS_MAPPING[name]
            
            # 获取边界框
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # 转换为YOLO格式（归一化中心坐标和宽高）
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            # 确保坐标在0-1范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            box_width = max(0, min(1, box_width))
            box_height = max(0, min(1, box_height))
            
            yolo_lines.append(f'{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}')
        
        # 写入YOLO格式文件
        with open(output_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        
        return True
    except Exception as e:
        print(f'转换错误 {xml_file}: {str(e)}')
        return False

def process_dataset(file_list, image_output_dir, label_output_dir):
    """处理数据集，转换标注并复制图像"""
    success_count = 0
    failure_count = 0
    
    for file_name in tqdm(file_list, desc=f'处理 {os.path.basename(image_output_dir)}'):
        # XML和图像文件路径
        xml_path = os.path.join(ANNOTATIONS_DIR, f'{file_name}.xml')
        image_path = os.path.join(IMAGES_DIR, f'{file_name}.png')
        
        # 检查文件是否存在
        if not os.path.exists(xml_path):
            print(f'警告: {xml_path} 不存在')
            failure_count += 1
            continue
        
        if not os.path.exists(image_path):
            print(f'警告: {image_path} 不存在')
            failure_count += 1
            continue
        
        # 转换标注
        output_label_path = os.path.join(label_output_dir, f'{file_name}.txt')
        if convert_xml_to_yolo(xml_path, output_label_path):
            # 复制图像
            output_image_path = os.path.join(image_output_dir, f'{file_name}.png')
            shutil.copy2(image_path, output_image_path)
            success_count += 1
        else:
            failure_count += 1
    
    print(f'处理完成: {success_count} 成功, {failure_count} 失败')
    return success_count, failure_count

def create_yaml_config():
    """创建YOLO配置文件"""
    yaml_path = os.path.join(OUTPUT_DIR, 'rps_final.yaml')
    
    # 获取绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_path = os.path.join(project_root, 'data', 'yolo_rps_final', 'images', 'train').replace('\\', '/')
    val_path = os.path.join(project_root, 'data', 'yolo_rps_final', 'images', 'val').replace('\\', '/')
    test_path = os.path.join(project_root, 'data', 'yolo_rps_final', 'images', 'test').replace('\\', '/')
    
    yaml_content = f"""# 使用绝对路径避免路径解析问题
train: {train_path}
val: {val_path}
test: {test_path}

# 类别数
nc: {len(CLASS_MAPPING)}

# 类别名称（A=石头, B=布, E=剪刀）
names: {list(CLASS_MAPPING.keys())}
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f'创建配置文件: {yaml_path}')
    return yaml_path

def main():
    print('开始CCO到YOLO格式转换...')
    
    # 创建目录
    create_directories()
    
    # 读取数据集列表
    train_files = read_list_file(TRAIN_LIST_PATH)
    val_files = read_list_file(VAL_LIST_PATH)
    test_files = read_list_file(TEST_LIST_PATH)
    
    print(f'训练集: {len(train_files)} 个样本')
    print(f'验证集: {len(val_files)} 个样本')
    print(f'测试集: {len(test_files)} 个样本')
    
    # 处理各个数据集
    train_success, train_failure = process_dataset(train_files, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN)
    val_success, val_failure = process_dataset(val_files, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL)
    test_success, test_failure = process_dataset(test_files, OUTPUT_IMAGES_TEST, OUTPUT_LABELS_TEST)
    
    # 创建配置文件
    yaml_path = create_yaml_config()
    
    total_success = train_success + val_success + test_success
    total_failure = train_failure + val_failure + test_failure
    
    print('\n转换总结:')
    print(f'总计: {total_success} 成功, {total_failure} 失败')
    print(f'配置文件: {yaml_path}')
    print('转换完成!')

if __name__ == '__main__':
    main()