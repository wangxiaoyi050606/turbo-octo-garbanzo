#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查训练好的模型的类别配置"""

from ultralytics import YOLO
import os

# 加载模型
model_path = os.path.join("..", "models", "output", "rps_final_model", "train", "weights", "best.pt")
model = YOLO(model_path)

print("=" * 50)
print("模型信息检查")
print("=" * 50)
print(f"模型路径: {model_path}")
print(f"类别数量: {len(model.names)}")
print(f"类别映射: {model.names}")
print("=" * 50)
