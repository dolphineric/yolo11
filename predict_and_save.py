import sys, os
# 將 ultralytics repo 根目錄加入 sys.path，讓 Python 能正確找到 ultralytics 套件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

from ultralytics import YOLO

def main(model_):
    # 1. 載入您訓練好的最佳權重 (請替換為您想測試的模型路徑)
    model = YOLO("runs/detect/train-5/weights/best.pt")  # 替換為您的模型路徑

    # 2. 指定要預測的圖片來源 (可以是資料夾、單張圖片、甚至是影片)
    # 例如：'CC_CLM/local_data/images/test'
    source_path = 'd:\\Dolphin\\NCYU\\citrus disease\\dataset\\CC_CLM\\local_data\\images\\test' 

    print(f"🚀 開始對 {source_path} 進行預測...")

    # 3. 執行預測
    results = model.predict(
        source=source_path,
        name= model_,
        project='predict/CC_CLM/result_20260518/' + model_,  # 預測結果將儲存在這個資料夾下
        save=True,        # 儲存畫上預測框的圖片 (方便肉眼比對結果)
        save_txt=True,    # 🌟 關鍵參數：自動生成 YOLO 格式的 .txt 標註檔
        save_conf=True,   # 🌟 實用推薦：在 txt 檔的最後面加上信心分數 (Confidence)
        conf=0.25,        # 信心門檻值：只儲存信心分數大於 0.25 的預測結果 (可依需求調整)
        device=0          # 使用 GPU 加速
    )

    print("\n✅ 預測與輸出完成！")
    print("👉 請前往 runs/detect/predict/labels/ 目錄下查看生成的 txt 檔案")

if __name__ == '__main__':
    list_of_files = [
        'yolo11-cbaca',
        #'yolo26s_exp',
        #'yolov8s_exp',
        #'yolov10s_exp'
        ]# 您可以在這裡添加更多模型路徑，或是從資料夾中讀取所有模型檔案
    for file in list_of_files:
        main(file)