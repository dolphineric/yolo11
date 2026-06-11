"""標準 YOLO11 訓練入口腳本（可指定資料集路徑）。

使用方式（在專案根目錄）：
    # 1) 不帶參數 → 預設用 data/citrus.yaml
    python ultralytics_yolo11\\train.py

    # 2) 帶上你的資料集 yaml（相對或絕對路徑都可）
    python ultralytics_yolo11\\train.py data/citrus.yaml
    python ultralytics_yolo11\\train.py D:\\path\\to\\my_data.yaml

    # 3) 進一步調整超參數
    python ultralytics_yolo11\\train.py data/citrus.yaml --epochs 200 --batch 8 --imgsz 640

會用 ultralytics_yolo11 這個精簡套件，以「標準 YOLO11 架構」訓練你指定的資料集，
結果存到 runs/yolo11/<name>/。
"""

import argparse
import os
import sys

# 專案根目錄 = ultralytics_yolo11 的上一層。先插入 sys.path，
# 不論從哪個工作目錄執行都能正確 import ultralytics_yolo11。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from ultralytics_yolo11 import YOLO  # noqa: E402

# 若要載入「原版 ultralytics 產生」的 .pt（pickle 路徑為 ultralytics.*），
# 取消下一行註解即可建立相容別名：
# sys.modules.setdefault("ultralytics", sys.modules["ultralytics_yolo11"])

# 預設資料集（你的柑橘資料集）
DEFAULT_DATA = os.path.join(ROOT, "data", "citrus.yaml")
# 標準 YOLO11 架構設定檔（n scale；可換 yolo11s/m/l/x.yaml）
DEFAULT_MODEL = os.path.join(ROOT, "ultralytics_yolo11", "cfg", "models", "11", "yolo11.yaml")


def parse_args():
    """解析命令列參數。第一個位置參數就是你的資料集 yaml 路徑。"""
    p = argparse.ArgumentParser(description="Train standard YOLO11 on a given dataset yaml.")
    p.add_argument(
        "data",
        nargs="?",
        default=DEFAULT_DATA,
        help="資料集設定檔路徑（.yaml）。不給就用 data/citrus.yaml。",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="模型架構或權重（.yaml 或 .pt）")
    p.add_argument("--epochs", type=int, default=200, help="訓練回合數")
    p.add_argument("--imgsz", type=int, default=640, help="輸入影像尺寸")
    p.add_argument("--batch", type=int, default=16, help="批次大小（VRAM 不足調小）")
    p.add_argument("--device", default=None, help="0 / 0,1 / cpu。不給就自動偵測 GPU")
    p.add_argument("--workers", type=int, default=8, help="資料載入執行緒（Windows 卡住可設 0）")
    p.add_argument("--project", default=os.path.join(ROOT, "runs", "yolo11"), help="結果主資料夾")
    p.add_argument("--name", default="train", help="本次訓練子資料夾名稱")
    # CLM 改良開關（不給就用 default.yaml 預設＝標準行為）
    p.add_argument("--box-iou", default=None, help="box 損失 IoU 類型: ciou/siou/eiou/giou/diou/wiou/wiouv3/nwd")
    p.add_argument("--tal-topk", type=int, default=None, help="TaskAlignedAssigner top-k（預設 10）")
    p.add_argument("--tal-alpha", type=float, default=None, help="TAL 分類指數（預設 0.5）")
    p.add_argument("--tal-beta", type=float, default=None, help="TAL 定位指數（預設 6.0）")
    return p.parse_args()


def resolve_data(path):
    """把使用者輸入的資料路徑轉成可用路徑：絕對路徑照用；相對路徑先試目前工作目錄，再試專案根目錄。"""
    if os.path.isabs(path):
        return path
    cand_cwd = os.path.abspath(path)
    if os.path.exists(cand_cwd):
        return cand_cwd
    cand_root = os.path.join(ROOT, path)
    if os.path.exists(cand_root):
        return cand_root
    return cand_cwd  # 交給 ultralytics 報明確的「找不到檔案」錯誤


def main():
    args = parse_args()
    data = resolve_data(args.data)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    print(f" 模型架構: {args.model}")
    print(f" 資料集  : {data}")
    print(f" device  : {device}")

    # 只把有給的 CLM 開關併入訓練參數（其餘走 default.yaml 預設）
    extra = {}
    for key, val in (
        ("box_iou", args.box_iou),
        ("tal_topk", args.tal_topk),
        ("tal_alpha", args.tal_alpha),
        ("tal_beta", args.tal_beta),
    ):
        if val is not None:
            extra[key] = val
    if extra:
        print(f" CLM 開關 : {extra}")

    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        **extra,
    )


if __name__ == "__main__":
    main()
