import sys, os
# 將 ultralytics repo 根目錄加入 sys.path，讓 Python 能正確找到 ultralytics 套件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11-CBACA.yaml")
model.train(data="data/citrus.yaml", epochs=100, imgsz=640, batch=4, workers=0, device=0, deterministic=False)

results = model.train(
    data='data/citrus.yaml',       # 資料集設定檔路徑
    epochs=300,               # 總訓練回合數
    imgsz=640,                # 圖片輸入尺寸
    batch=16,                 # 批次大小 
    patience=40,              # Early Stopping 耐心值
    device=0,                 # 指定使用 GPU 
    project='CC_CLM/result_20260428',  # 訓練結果存檔的主資料夾
    name= "yolo11-CBACA" +'_exp',  # 這次訓練的子資料夾名稱
    # ==========================================
    # 🛑 關閉所有主要的資料擴增參數
    # ==========================================
    
    #mosaic=0.0,      # 關閉馬賽克拼接 (預設 1.0)
    #mixup=0.0,       # 關閉圖片疊影混合 (預設 0.0，確認關閉)
    #hsv_h=0.0,       # 關閉色調隨機變化 (預設 0.015)
    #hsv_s=0.0,       # 關閉飽和度隨機變化 (預設 0.7)
    #hsv_v=0.0,       # 關閉明度隨機變化 (預設 0.4)
    #degrees=0.0,     # 關閉隨機旋轉 (預設 0.0)
    #translate=0.0,   # 關閉隨機平移 (預設 0.1)
    #scale=0.0,       # 關閉隨機縮放 (預設 0.5)
    #shear=0.0,       # 關閉隨機錯切變形 (預設 0.0)
    #perspective=0.0, # 關閉隨機透視變形 (預設 0.0)
    #fliplr=0.0,      # 關閉左右隨機翻轉 (預設 0.5)
    #flipud=0.0,      # 關閉上下隨機翻轉 (預設 0.0)
    #erasing=0.0      # 關閉隨機遮擋/擦除 (預設 0.4)
    
    workers=8,        # 加速資料載入
    cache=True,       # 將圖片放入記憶體 (關鍵!)
)