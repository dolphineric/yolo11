import os
import sys
from pathlib import Path

# 改成你 clone 下來的 ultralytics_pro 路徑
PROJECT_DIR = r"D:\Dolphin\NCYU\yolo\ultralytics"

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

print("目前工作目錄：", os.getcwd())