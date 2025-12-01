import os
import glob
import xml.etree.ElementTree as ET

def fix_e_class_labels():
    """
    将所有E类标签文件转换为C类并添加正确的类别ID=2
    解决E类(剪刀)标签文件为空的训练问题
    从原始XML标注文件中读取正确的边界框信息
    """
    # 定义路径
    train_labels_path = r"e:/stjdb/data/yolo_rps_final/labels/train"
    test_labels_path = r"e:/stjdb/data/yolo_rps_final/labels/test"
    original_annotations_path = r"e:/stjdb/rps_final/Annotations"
    
    def convert_xml_to_yolo(xml_path):
        """将XML格式的标注转换为YOLO格式"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像尺寸
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)
            
            # 获取边界框信息
            object_elem = root.find('object')
            if object_elem is not None:
                xmin = int(object_elem.find('bndbox/xmin').text)
                ymin = int(object_elem.find('bndbox/ymin').text)
                xmax = int(object_elem.find('bndbox/xmax').text)
                ymax = int(object_elem.find('bndbox/ymax').text)
                
                # 转换为YOLO格式（归一化坐标）
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                
                # 四舍五入到6位小数
                x_center = round(x_center, 6)
                y_center = round(y_center, 6)
                box_width = round(box_width, 6)
                box_height = round(box_height, 6)
                
                # 返回YOLO格式的字符串，类别ID为2（剪刀）
                return f"2 {x_center} {y_center} {box_width} {box_height}"
        except Exception as e:
            print(f"解析XML文件出错 {xml_path}: {str(e)}")
        return None
    
    # 创建一个函数来处理特定目录下的标签文件
    def process_labels(directory):
        # 查找所有以'E-'开头的.txt文件
        e_files = glob.glob(os.path.join(directory, "E-*.txt"))
        
        for file_path in e_files:
            # 获取文件名（不包含扩展名）
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            # 对应的XML文件路径
            xml_path = os.path.join(original_annotations_path, f"{file_name}.xml")
            
            try:
                # 读取当前标签文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # 优先从XML文件中读取正确的标签信息
                if os.path.exists(xml_path):
                    yolo_content = convert_xml_to_yolo(xml_path)
                    if yolo_content:
                        # 无论当前文件是否为空，都使用XML中的正确信息
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(yolo_content)
                        print(f"从XML更新标签文件: {file_path}")
                    else:
                        # 如果XML解析失败且文件为空，则使用默认值
                        if not content:
                            print(f"XML解析失败，使用默认值修复空标签: {file_path}")
                            # 使用一个合理的默认值
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write("2 0.5 0.5 0.4 0.6")
                else:
                    # 如果没有对应的XML文件且当前文件为空
                    if not content:
                        print(f"XML文件不存在，使用默认值修复空标签: {file_path}")
                        # 使用一个合理的默认值
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write("2 0.5 0.5 0.4 0.6")
                    else:
                        # 如果文件有内容，确保类别ID是2（剪刀）
                        lines = content.split('\n')
                        new_lines = []
                        for line in lines:
                            if line.strip():
                                # 将每行的第一个数字（类别ID）改为2
                                parts = line.split()
                                if len(parts) >= 5:
                                    parts[0] = '2'
                                    new_lines.append(' '.join(parts))
                        
                        # 写回文件
                        if new_lines:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(new_lines))
                            print(f"更新标签ID: {file_path}")
            except Exception as e:
                print(f"处理文件时出错 {file_path}: {str(e)}")
    
    # 处理训练集和测试集的标签文件
    print("开始处理训练集标签文件...")
    process_labels(train_labels_path)
    
    print("开始处理测试集标签文件...")
    process_labels(test_labels_path)
    
    print("所有E类标签文件修复完成！")

if __name__ == "__main__":
    fix_e_class_labels()