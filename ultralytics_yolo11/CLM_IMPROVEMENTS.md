# YOLO11 CLM 改良說明（不規則 / 大小不一病徵）

針對潛葉蛾(CLM)等「形狀不規則、大小不一」病徵，新增四項可組合、可切換、**向後相容**的改良。
不傳任何新參數時，行為與標準 YOLO11 完全相同。全部已用 1-epoch 冒煙訓練 + 驗證 + 預測驗證通過。

## A. 偵測尺度 / Neck — 新增 P2 小物件頭
新 yaml（`cfg/models/11/`），`Detect` 偵測 **P2/4、P3/8、P4/16、P5/32** 四尺度（多了 stride 4 接小目標）：

| yaml | 內容 |
|---|---|
| `yolo11-p2.yaml` | 標準 backbone + P2 頭（純尺度改良，做為對照基準） |
| `yolo11-clm-dcnb.yaml` | P2 頭 + **backbone 深層 P4/P5 用可變形卷積** |
| `yolo11-clm-dcnn.yaml` | P2 頭 + **neck 用可變形卷積** |

## B. 形狀自適應卷積 — Deformable Conv v2（`C3k2_DCN`）
`nn/modules/block.py` 新增 `DCNv2 / Bottleneck_DCN / C3k2_DCN`（用 torchvision `DeformConv2d`），
採樣點隨病徵幾何變形。`C3k2_DCN` 與 `C3k2` 介面相同，可在任何 yaml 直接替換。

## C. 可切換 box 損失（對歧義/不規則框更穩健）
`--box-iou` 或 `model.train(box_iou=...)`，可選：
`ciou`(預設) / `siou` / `eiou` / `giou` / `diou` / `wiou` / `wiouv3` / `nwd`
（`nwd` = Normalized Wasserstein Distance，小目標友善，給 canker 重用。）

## D. 可調標籤分配（TaskAlignedAssigner）
`--tal-topk`(預設10) / `--tal-alpha`(預設0.5) / `--tal-beta`(預設6.0)。

## 使用方式（在專案根目錄，系統 Python39）
```powershell
$py = "C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe"

# 只加 P2 頭（A，對照基準）
& $py ultralytics_yolo11\train.py data/citrus.yaml --model ultralytics_yolo11/cfg/models/11/yolo11-p2.yaml

# P2 + backbone DCN + SIoU 損失（A+B+C）
& $py ultralytics_yolo11\train.py data/citrus.yaml `
    --model ultralytics_yolo11/cfg/models/11/yolo11-clm-dcnb.yaml --box-iou siou

# 標準架構，只調損失與分配（C+D，最便宜的對照）
& $py ultralytics_yolo11\train.py data/citrus.yaml --box-iou wiouv3 --tal-beta 4.0 --tal-topk 13
```

## 建議的 ablation 順序
1. `yolo11.yaml`（基準）→ 2. `yolo11-p2.yaml`（加 A）→ 3. `+ --box-iou siou/wiouv3`（加 C）
→ 4. `yolo11-clm-dcnb/dcnn.yaml`（加 B）→ 5. 微調 `--tal-*`（D）。
一次只改一項、固定 `seed`，同時看 **mAP50 與 mAP50-95**。

## 注意事項
- **DCN 與 `deterministic`**：`deform_conv2d` 的反向沒有確定性核心；含 DCN 的模型第一次 forward 會自動把
  `torch.use_deterministic_algorithms` 放寬為 `warn_only`（仍盡量確定性，只是不再 error）。這是預期行為。
- DCN 在 AMP 下會以 fp32 執行該層（torchvision 無 autocast 核心），略增記憶體/時間，正確性優先。
- DCN 版 VRAM 與耗時較高；RTX 3050 若 OOM 請把 `--batch` 調小或降 `imgsz`。
- 這些開關的預設值＝標準行為，所以 `python ultralytics_yolo11\train.py`（不帶新參數）與改良前完全一致。

---

# Canker(潰瘍病) — 小病徵 / 模糊邊界

Canker 病徵很小、IoU 對小框位移極敏感（P/R 易崩），且邊界模糊不易精準切邊。
新增 `cfg/models/11/yolo11-canker.yaml` = **四尺度 P2 頭（P2/P3/P4/P5）+ `reg_max: 24`**，搭配既有 `--box-iou nwd` 與 recipe；**無核心程式碼改動**。

## 訓練 recipe
```powershell
$py = "C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe"
& $py ultralytics_yolo11\train.py data/citrus.yaml `
    --model ultralytics_yolo11/cfg/models/11/yolo11-canker.yaml `
    --box-iou nwd --imgsz 1280 --batch 4
```
- **`--box-iou nwd`**：把框當 2D 高斯算距離相似度，對小框位移遠比 IoU 寬容（小目標 P/R 的關鍵）。常數預設 12.7（小目標標準值）；需微調改 `utils/metrics.py` 的 `bbox_nwd(constant=...)`。
- **`--imgsz 1280`**：小病徵像素變多，通常比任何模組改動更直接（VRAM 不足就降 `--batch` 或 imgsz）。
- 擴增別把小目標縮更小：降 `scale`、必要時提早 `close_mosaic`；只有一類可加 `single_cls=True`。
- **評估**：小目標看 **mAP50**（mAP50-95 天然偏低）；報告 P/R 請取 **best-F1 對應的 conf**，不要隨手用 0.25。
- 進一步（未實作，建議）：**SAHI 切片訓練/推論**對小病徵常是決定性的。

## 模糊邊界 — 模型能怎麼輔助？
1. **DFL + 高 `reg_max`（已內建於 canker yaml）**：YOLO11 的框迴歸不是回歸硬數值，而是每條邊學一個**機率分佈**再取期望；邊界模糊時分佈會自然變寬表達不確定性。`reg_max` 16→24 給更細的邊位置解析度。要回標準改回 16 即可。
2. **NWD 高斯框損失**（`--box-iou nwd`）：對不精準的邊界與小框都更寬容。
3. **標註慣例 > 切邊精準度**：邊界模糊時，定一個**一致**規則（例如「可見病變外緣」）整批套用，殘差交給 DFL 吸收；最忌時鬆時緊、漏標。
4. **偵測本就比分割寬容**：bbox 不需像素級邊界；除非要表達區域，否則不必為了模糊邊界改用分割。
5. 可選：分類加 label smoothing、推論用 TTA（`augment=True`）。

> 註：`reg_max:24` 會讓 Detect 框分支輸出 4×24=96 通道（已驗證 `nl=4`、train(NWD)→predict 端到端跑通）。
