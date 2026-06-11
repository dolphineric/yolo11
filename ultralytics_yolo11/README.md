# ultralytics_yolo11

從 `ultralytics`（v8.4.48，客製版）精簡而來的**獨立 YOLO11 套件**，只保留標準 YOLO11
物件偵測的訓練 / 推論所需程式，並把套件改名為 `ultralytics_yolo11`，與原本的
`ultralytics` 並存不衝突。

- 已移除與 YOLO11 無關的大型模型家族：**SAM / SAM3 / RT-DETR / NAS / FastSAM**。
- 保留完整 YOLO 偵測流程（含 nn / engine / data / utils / cfg 與 YOLO 各任務）。
- 所有內部 `import ultralytics.*` 都已改為 `import ultralytics_yolo11.*`。

## 使用方式（在專案根目錄 `D:\Dolphin\NCYU\yolo` 執行）

請使用**有安裝套件的 Python**（系統 Python 3.9，內含 torch+CUDA / opencv 等）：

```powershell
# 訓練（標準 YOLO11，data/citrus.yaml，結果存到 runs/yolo11/train）
C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe ultralytics_yolo11\train.py

# 推論（載入 runs/yolo11/train/weights/best.pt，預測 data/images/test）
C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe ultralytics_yolo11\predict.py
```

> 注意：專案根目錄下的 `.venv` 沒有安裝任何套件，請勿用它執行。

- `train.py`：可改 `epochs` / `imgsz` / `batch`；VRAM 不足時把 `batch` 調小。
  想換大小可把 `yolo11.yaml` 改成 `yolo11s/m/l/x.yaml`。
- `predict.py`：請先完成訓練產生 `best.pt`，再執行；可改 `source` / `conf`。
- `device` 會自動偵測 GPU，沒有 GPU 時退回 CPU。

## 程式來源對照（YOLO11 架構用到的主要函式/模組）

| 功能 | 位置 |
| --- | --- |
| 模型設定檔（標準架構） | `cfg/models/11/yolo11.yaml` |
| 模型建構 / 解析 | `nn/tasks.py`（`parse_model`、`DetectionModel`） |
| 基礎模組 | `nn/modules/conv.py`、`block.py`、`head.py`（Conv / C3k2 / SPPF / C2PSA / Concat / Detect…） |
| 訓練 / 推論引擎 | `engine/trainer.py`、`model.py`、`predictor.py`、`validator.py`、`results.py` |
| YOLO 偵測任務 | `models/yolo/detect/{train,val,predict}.py` |
| 資料載入 / 擴增 | `data/*` |
| 通用工具 | `utils/*` |

## 已知限制 / 備註

- **AMP 數值預檢**會嘗試載入官方 `yolo26n.pt` 作參考；在本精簡套件中此步驟會被
  安全略過（訓練照常進行、AMP 仍啟用），這是正常現象。
- 本套件存出的 `.pt` 內部類別路徑是 `ultralytics_yolo11.*`，自家 train → predict
  流程完全相容。若要載入**原版 ultralytics 產生**的 `.pt`（類別路徑為
  `ultralytics.*`），請在腳本最前面加：
  `import sys; sys.modules.setdefault("ultralytics", sys.modules["ultralytics_yolo11"])`
- 超參數搜尋 `model.tune()` 未在此精簡範圍內驗證（其子行程路徑仍指向 `ultralytics`）。
