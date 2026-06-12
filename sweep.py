"""自動掃參腳本：把 model × box_iou × tal_beta × imgsz 排成矩陣，依序訓練，
並把每個 run 的「最佳 epoch」指標收集到 runs/sweep/summary.csv，最後印出排序表。

特點
- 可重跑（resume）：summary.csv 裡已完成的組合會自動略過。
- 容錯：某個組合失敗（例如 OOM）會記成 FAIL 並繼續下一個，不會整批中斷。
- 結果即時寫檔：每跑完一個就寫一行，中途斷電也不會白跑。

用法（在專案根目錄，系統 Python39）：
  C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python39\\python.exe sweep.py
  ... sweep.py --epochs 100 --batch 4 --device 0          # 篩選階段用少 epoch 排名，贏家再全訓
  ... sweep.py --data data/citrus.yaml --workers 2

要改「掃哪些值」→ 直接編輯下面的 GRID 區塊。
"""

import argparse
import csv
import gc
import itertools
import os
import sys
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import logging  # noqa: E402

import torch  # noqa: E402
from ultralytics_yolo11 import YOLO  # noqa: E402

# ── 修正：防止 "ultralytics" logger 被兩個套件各加一個 handler，導致每行訊息重複顯示兩次 ──
_ul_logger = logging.getLogger("ultralytics")
if len(_ul_logger.handlers) > 1:
    # 保留第一個 handler，移除多餘的
    for _h in _ul_logger.handlers[1:]:
        _ul_logger.removeHandler(_h)

# ============================ 要掃的矩陣（改這裡）============================
MODELS = [
    "yolo11s-clm-dcnn.yaml",   # 目前最佳；可加 "yolo11s.yaml", "yolo11s-canker.yaml" ...
]
BOX_IOU = ["ciou", "siou", "nwd"]   # ciou(基準) / siou(不規則) / eiou / wiouv3 / nwd(小目標)
TAL_BETA = [6.0, 4.0]               # 6.0=預設；小目標可降到 4.0 / 2.0
IMGSZ = [640]                       # 想做解析度掃描就加 768, 960

# 固定不變的覆寫（所有組合共用；要調 tal_alpha/topk、optimizer 等放這）
BASE = dict(
    tal_alpha=0.5,
    tal_topk=10,
    # optimizer="SGD", cos_lr=True, label_smoothing=0.0,
)
# ===========================================================================

CFG_DIR = os.path.join(ROOT, "ultralytics_yolo11", "cfg", "models", "11")


def parse_best(results_csv):
    """從 results.csv 取『mAP50-95 最高那一列』的指標。回傳 dict 或 None。"""
    if not os.path.exists(results_csv):
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(results_csv))]
    if not rows:
        return None

    def g(row, key):
        for k in row:
            if k.replace(" ", "") == key:
                try:
                    return float(row[k])
                except Exception:
                    return None
        return None

    best = max(rows, key=lambda x: (g(x, "metrics/mAP50-95(B)") or -1))
    return dict(
        mAP50=g(best, "metrics/mAP50(B)") or 0.0,
        mAP5095=g(best, "metrics/mAP50-95(B)") or 0.0,
        P=g(best, "metrics/precision(B)") or 0.0,
        R=g(best, "metrics/recall(B)") or 0.0,
        bestEp=int(g(best, "epoch") or 0),
        ranEp=len(rows),
    )


SUMMARY_FIELDS = ["name", "model", "box_iou", "tal_beta", "imgsz",
                  "mAP50", "mAP5095", "P", "R", "bestEp", "ranEp", "status", "time"]


def load_done(summary_csv):
    done = set()
    if os.path.exists(summary_csv):
        for r in csv.DictReader(open(summary_csv)):
            if r.get("status") == "OK":
                done.add(r["name"])
    return done


def append_summary(summary_csv, row):
    new = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="YOLO11 hyperparameter sweep")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "citrus.yaml"))
    ap.add_argument("--epochs", type=int, default=100, help="篩選階段建議 80~100，贏家再全訓")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2, help="Windows/RAM 吃緊建議 0~4")
    ap.add_argument("--device", default=None, help="0 / cpu；不給自動偵測")
    ap.add_argument("--project", default=os.path.join(ROOT, "runs", "sweep"))
    ap.add_argument("--fraction", type=float, default=1.0, help="只用部分資料快篩（如 0.2）")
    args = ap.parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    summary_csv = os.path.join(args.project, "summary.csv")
    os.makedirs(args.project, exist_ok=True)
    done = load_done(summary_csv)

    combos = list(itertools.product(MODELS, BOX_IOU, TAL_BETA, IMGSZ))
    print(f"共 {len(combos)} 個組合；已完成 {len(done)} 個會略過。device={device} epochs={args.epochs}")
    print("=" * 70)

    for i, (model, biou, beta, sz) in enumerate(combos, 1):
        mtag = os.path.splitext(os.path.basename(model))[0]
        name = f"{mtag}_{biou}_b{beta:g}_sz{sz}"
        if name in done:
            print(f"[{i}/{len(combos)}] 略過（已完成）: {name}")
            continue

        print(f"\n[{i}/{len(combos)}] 訓練: {name}")
        extra = dict(BASE)
        extra.update(box_iou=biou, tal_beta=beta)
        row = dict(name=name, model=mtag, box_iou=biou, tal_beta=beta, imgsz=sz,
                   mAP50="", mAP5095="", P="", R="", bestEp="", ranEp="",
                   status="FAIL", time=datetime.now().strftime("%Y-%m-%d %H:%M"))
        try:
            m = YOLO(os.path.join(CFG_DIR, model))
            m.train(
                data=args.data, epochs=args.epochs, imgsz=sz, batch=args.batch,
                device=device, workers=args.workers, project=args.project, name=name,
                exist_ok=True, fraction=args.fraction, **extra,
            )
            best = parse_best(os.path.join(str(m.trainer.save_dir), "results.csv"))
            if best:
                row.update(status="OK", **{k: round(v, 4) if isinstance(v, float) else v
                                           for k, v in best.items()})
                print(f"   [OK] mAP50={best['mAP50']:.3f}  mAP50-95={best['mAP5095']:.3f}  "
                      f"P={best['P']:.3f}  R={best['R']:.3f}  (bestEp={best['bestEp']})")
        except Exception as e:
            traceback.print_exc()
            print(f"   [FAIL] {name}：{e}")
        finally:
            append_summary(summary_csv, row)
            try:
                del m
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 最後印排序表
    print("\n" + "=" * 70)
    print("結果排序（依 mAP50-95 由高到低）：", summary_csv)
    rows = [r for r in csv.DictReader(open(summary_csv)) if r.get("status") == "OK"]
    rows.sort(key=lambda r: float(r["mAP5095"] or 0), reverse=True)
    print("{:36s}{:>8}{:>9}{:>7}{:>7}".format("name", "mAP50", "mAP5095", "P", "R"))
    for r in rows:
        print("{:36s}{:>8}{:>9}{:>7}{:>7}".format(
            r["name"], r["mAP50"], r["mAP5095"], r["P"], r["R"]))


if __name__ == "__main__":
    main()
