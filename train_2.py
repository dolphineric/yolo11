from ultralytics import YOLO
import datetime # 🌟 新增：用來獲取當前時間

def main(model_name):
    print(f"\n🚀 開始載入 {model_name} 模型...")
    
    # 1. 載入預訓練模型
    model = YOLO(model_name + ".pt") 

    print("⏳ 開始進行模型訓練...")

    # 2. 啟動訓練
    results = model.train(
        data='citrus.yaml',       # 資料集設定檔路徑
        epochs=300,               # 總訓練回合數
        imgsz=640,                # 圖片輸入尺寸
        batch=16,                 # 批次大小 
        patience=40,              # Early Stopping 耐心值
        device=0,                 # 指定使用 GPU 
        project='CC_CLM/result_20260428',  # 訓練結果存檔的主資料夾
        name= model_name +'_exp',  # 這次訓練的子資料夾名稱
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

    # 3. 🌟 新增：提取最佳指標並附加寫入 TXT 總表
    print("\n📊 正在提取最佳驗證成績並存檔...")
    precision = results.box.mp
    recall = results.box.mr
    map50 = results.box.map50
    map50_95 = results.box.map

    # 準備該模型的文字報表
    report_text = (
        f"🟢 模型名稱: {model_name}\n"
        f"   - Precision : {precision:.4f}\n"
        f"   - Recall    : {recall:.4f}\n"
        f"   - mAP@50    : {map50:.4f}\n"
        f"   - mAP@50-95 : {map50_95:.4f}\n"
        f"-"*40 + "\n"
    )

    # 使用 'a' (附加模式) 寫入，確保接在舊資料後面不會覆蓋
    with open('citrus_training_report.txt', 'a', encoding='utf-8') as f:
        f.write(report_text)

    print("✅ 訓練任務與成績記錄結束！")
    # 🌟 修改提示訊息：對應你設定的 project 與 name 路徑
    print(f"👉 最佳權重與訓練圖表已儲存至： CC_CLM/result/{model_name}_exp 目錄下")

# ⚠️ 重要防呆機制
if __name__ == '__main__':
    
    model_list = [
        # --- YOLO11 系列  ---
        'yolo11x',

        # --- YOLOv8 系列  ---
        'yolov8x',

        # --- YOLOv10 系列 ---
        'yolov10x',

        # --- YOLOv26 系列 ---
        'yolo26x'
    ]
    
    '''
    model_list = [
        # --- YOLOv26 系列 ---
        'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x', 'yolo11x'
    ]
    '''
    
    # 🌟 新增：在迴圈開始前，先建立一個新的總表並寫入標題與系統時間
    # 使用 'w' 模式，每次重新執行這個腳本時會刷新一份新總表
    with open('citrus_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"🍊 柑橘葉片辨識模型 - 訓練評估總表\n")
        f.write(f"建立時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*40 + "\n")

    # 執行迴圈，依序將陣列內的模型送入 main() 函數訓練
    for i in model_list:
        main(i)