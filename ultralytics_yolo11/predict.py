"""標準 YOLO11 推論入口腳本。

使用方式（在專案根目錄）：
    python ultralytics_yolo11\\predict.py

預設載入 train.py 產生的最佳權重 runs/yolo11/train/weights/best.pt，
對 data/images/test 進行預測，結果（畫框圖 + YOLO 格式 txt）存到
runs/yolo11/predict/。請先執行 train.py 產生權重，再執行本腳本。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from ultralytics_yolo11 import YOLO  # noqa: E402

# 若要載入「原版 ultralytics 產生」的 .pt，取消下一行註解：
# sys.modules.setdefault("ultralytics", sys.modules["ultralytics_yolo11"])


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    # 訓練完成後的最佳權重（可改成你要測試的任何 best.pt 路徑）
    weights = os.path.join(ROOT, "runs", "yolo11", "train", "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"找不到權重：{weights}\n請先執行 `python ultralytics_yolo11\\train.py` 完成訓練。"
        )

    model = YOLO(weights)
    model.predict(
        source=os.path.join(ROOT, "data", "images", "test"),  # 要預測的影像來源
        save=True,         # 存下畫了預測框的圖片
        save_txt=True,     # 輸出 YOLO 格式 .txt 標註
        save_conf=True,    # txt 內附上信心分數
        conf=0.25,         # 信心門檻
        device=device,
        project=os.path.join(ROOT, "runs", "yolo11"),
        name="predict",    # 結果資料夾：runs/yolo11/predict
    )


if __name__ == "__main__":
    main()
