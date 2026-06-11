# 標準 YOLO11 架構流程圖與函式分解

依據 `cfg/models/11/yolo11.yaml` 與 `nn/modules/{conv,block,head}.py` 實際程式整理。
（yaml 內的通道數是基準值；以 scale `n`=寬度 0.25 為例，會再乘上寬度係數。）

---

## 0. 模組分類（六大類）

| 類別 | 模組 | 作用 |
| --- | --- | --- |
| 下採樣 / Stem | `Conv`(stride=2) | 縮小解析度、加深通道 |
| 特徵萃取 (CSP) | `C3k2` →(`C3k`/`Bottleneck`) | 主要的卷積特徵學習 |
| 多尺度池化 | `SPPF` | 擴大感受野、聚合多尺度上下文 |
| 注意力 | `C2PSA` →(`PSABlock`→`Attention`) | 位置敏感自注意力 |
| 特徵融合 | `Concat`、`nn.Upsample` | 上採樣 + 跨層特徵拼接 (PAN-FPN) |
| 偵測頭 | `Detect` →(`Conv`/`DWConv`/`DFL`) | 輸出框迴歸 + 類別 |

---

## 1. 整體資料流：影像 → Backbone → Neck → Head

```mermaid
graph TD
    IMG["影像 (3, 640, 640)"] --> B

    subgraph B["Backbone（萃取多尺度特徵）"]
      L0["0 Conv s2  P1/2"] --> L1["1 Conv s2  P2/4"]
      L1 --> L2["2 C3k2"] --> L3["3 Conv s2  P3/8"]
      L3 --> L4["4 C3k2  ★P3"] --> L5["5 Conv s2  P4/16"]
      L5 --> L6["6 C3k2  ★P4"] --> L7["7 Conv s2  P5/32"]
      L7 --> L8["8 C3k2"] --> L9["9 SPPF"] --> L10["10 C2PSA  ★P5"]
    end

    subgraph N["Neck（PAN-FPN：上採樣+下採樣雙向融合）"]
      L11["11 Upsample"] --> L12["12 Concat(+P4)"] --> L13["13 C3k2"]
      L13 --> L14["14 Upsample"] --> L15["15 Concat(+P3)"] --> L16["16 C3k2 →小物件"]
      L16 --> L17["17 Conv s2"] --> L18["18 Concat(+13)"] --> L19["19 C3k2 →中物件"]
      L19 --> L20["20 Conv s2"] --> L21["21 Concat(+P5)"] --> L22["22 C3k2 →大物件"]
    end

    L10 --> L11
    L4  -. P3 .-> L15
    L6  -. P4 .-> L12
    L10 -. P5 .-> L21
    L13 -. .-> L18

    L16 --> H["23 Detect"]
    L19 --> H
    L22 --> H
    H --> OUT["輸出：每格 (4·reg_max 框分佈 + nc 類別)"]
```

### 逐層對照表

| # | from | 模組 | 解析度 | 角色 |
|---|------|------|--------|------|
| 0 | img | Conv s2 | P1/2 | stem 下採樣 |
| 1 | 0 | Conv s2 | P2/4 | 下採樣 |
| 2 | 1 | C3k2 (Bottleneck) | P2/4 | 特徵 |
| 3 | 2 | Conv s2 | P3/8 | 下採樣 |
| 4 | 3 | C3k2 (Bottleneck) | **P3/8 ★** | 特徵（→第15層融合） |
| 5 | 4 | Conv s2 | P4/16 | 下採樣 |
| 6 | 5 | C3k2 (C3k) | **P4/16 ★** | 特徵（→第12層融合） |
| 7 | 6 | Conv s2 | P5/32 | 下採樣 |
| 8 | 7 | C3k2 (C3k) | P5/32 | 特徵 |
| 9 | 8 | SPPF | P5/32 | 多尺度池化 |
| 10 | 9 | C2PSA | **P5/32 ★** | 注意力（→第21層融合） |
| 11 | 10 | nn.Upsample | P4 | 上採樣 ×2 |
| 12 | 11,6 | Concat | P4 | 融合 P5↑+P4 |
| 13 | 12 | C3k2 | P4 | 特徵 |
| 14 | 13 | nn.Upsample | P3 | 上採樣 ×2 |
| 15 | 14,4 | Concat | P3 | 融合 +P3 |
| 16 | 15 | C3k2 | **P3/8 小** | Detect 輸入 |
| 17 | 16 | Conv s2 | P4 | 下採樣 |
| 18 | 17,13 | Concat | P4 | 融合 |
| 19 | 18 | C3k2 | **P4/16 中** | Detect 輸入 |
| 20 | 19 | Conv s2 | P5 | 下採樣 |
| 21 | 20,10 | Concat | P5 | 融合 |
| 22 | 21 | C3k2 (C3k) | **P5/32 大** | Detect 輸入 |
| 23 | 16,19,22 | Detect | — | 偵測頭 |

---

## 2. 複合模組 → 最基礎函式 分解樹

> 規則：`Conv = nn.Conv2d(bias=False) → nn.BatchNorm2d → nn.SiLU`
> （若 `act=False`，最後一步換成 `nn.Identity`；`DWConv` = `Conv` 但 `groups=輸入通道`，即深度卷積。）

```
Conv(c1,c2,k,s)                      # nn/modules/conv.py
└─ nn.Conv2d → nn.BatchNorm2d → nn.SiLU

Bottleneck(shortcut)                 # block.py:458
└─ Conv(1×1或k) → Conv(k) → [ + x ]   (殘差相加，當 shortcut 且同通道)

C3k2  (c3k=False，第 2/4/13/16/19 層)  # block.py:1070，繼承 C2f
├─ cv1: Conv(1×1) → chunk(2)
├─ m  : n × Bottleneck
└─ torch.cat → cv2: Conv(1×1)

C3k2  (c3k=True，第 6/8/22 層)         # m 改用 C3k
└─ C3k → C3(block.py:323)
   ├─ cv1: Conv(1×1)          ┐
   ├─ cv2: Conv(1×1)          ├─ 兩條分支
   ├─ m  : n × Bottleneck(k=3)┘
   └─ torch.cat → cv3: Conv(1×1)

SPPF(k=5)                            # block.py:209
├─ cv1: Conv(1×1)
├─ m  : nn.MaxPool2d(5,1,2) 串接 3 次
└─ torch.cat(4 張) → cv2: Conv(1×1)

C2PSA                                # block.py:1485
├─ cv1: Conv(1×1) → split(2)  →  a, b
├─ m  : n × PSABlock(只作用在 b)
│   └─ PSABlock                      # block.py:1380
│      ├─ x + Attention(x)
│      │   └─ Attention              # block.py:1320
│      │      ├─ qkv: Conv(1×1)         → split → q,k,v
│      │      ├─ (qᵀ·k)·scale → softmax  (注意力權重)
│      │      ├─ v·attnᵀ  +  pe: Conv(3×3 深度卷積)
│      │      └─ proj: Conv(1×1)
│      └─ x + FFN(x)
│          └─ FFN: Conv(1×1) → Conv(1×1, 無激活)
└─ torch.cat(a,b) → cv2: Conv(1×1)

Concat                               # conv.py:757
└─ torch.cat

nn.Upsample(scale=2, 'nearest')      # = F.interpolate（最近鄰）

Detect(nc, ch=[P3,P4,P5])            # head.py:26（每個尺度各一組）
├─ cv2 框分支: Conv(3×3) → Conv(3×3) → nn.Conv2d(1×1)   → 4·reg_max
├─ cv3 類分支: [DWConv(3×3)→Conv(1×1)]×2 → nn.Conv2d(1×1) → nc
├─ 每層輸出 torch.cat(框, 類)
└─ 推論時：make_anchors → DFL → dist2bbox → torch.sigmoid(類別)
   └─ DFL(block.py:59): view/transpose → softmax(reg_max) → nn.Conv2d(固定權重) 求期望
```

---

## 3. 「最基礎函式」總表（依類別分類）

整個標準 YOLO11 最終都歸結到以下原子操作（leaf functions）：

### A. 卷積
| 函式 | 說明 | 出現於 |
|------|------|--------|
| `nn.Conv2d` (bias=False) | 標準卷積：1×1、3×3、stride-2 下採樣 | Conv、所有模組 |
| `nn.Conv2d` (groups=in) | 深度卷積 (DWConv、Attention.pe) | Detect.cv3、Attention |
| `nn.Conv2d` (固定權重, 不更新) | DFL 積分（權重=0,1,…,reg_max-1） | Detect.DFL |

### B. 正規化
| `nn.BatchNorm2d` | 批次正規化，接在每個 Conv 後 | 所有 Conv |

### C. 激活 / 機率化
| `nn.SiLU` | 預設激活 `x·sigmoid(x)` | 所有 Conv (act=True) |
| `nn.Identity` | `act=False` 時的佔位（不激活） | qkv/pe/proj、SPPF.cv1… |
| `.softmax(dim)` | 注意力權重、DFL 分佈 | Attention、DFL |
| `torch.sigmoid` | 推論時類別分數 | Detect（推論） |

### D. 池化 / 取樣
| `nn.MaxPool2d(5,1,2)` | SPPF 滑動最大池化（重複 3 次≈SPP 5/9/13） | SPPF |
| `nn.Upsample` / `F.interpolate('nearest')` | 特徵圖放大 ×2 | Neck 第 11/14 層 |

### E. 張量重組
| `torch.cat` | 沿通道拼接特徵 | Concat、C2f/C3、SPPF、C2PSA、Detect |
| `.chunk` / `.split` | 沿通道切分（CSP 分流 / qkv 分流） | C2f、C2PSA、Attention |
| `.view` / `.reshape` / `.transpose` | 形狀重排（注意力、DFL） | Attention、DFL、Detect |

### F. 元素級運算
| `+`（殘差相加） | 跳接，緩解梯度消失 | Bottleneck、PSABlock |
| `@`（`torch.matmul`） | 注意力 q·k、attn·v | Attention |
| `× scale` | 注意力分數縮放 `key_dim^-0.5` | Attention |

### G. 偵測後處理（`utils/tal.py`）
| `make_anchors` | 由各尺度特徵圖產生 anchor 中心與 stride | Detect（推論） |
| `dist2bbox` | 把 DFL 預測的四方向距離轉成 xyxy/xywh 框 | Detect（推論） |

---

### 一句話總結
標準 YOLO11 = 大量 **`Conv2d + BatchNorm2d + SiLU`**（基本卷積）堆疊成
**CSP (C3k2)** 萃取特徵，輔以 **SPPF（MaxPool）** 擴大感受野、**C2PSA（matmul+softmax 自注意力）**
強化語意，再用 **Upsample + Concat** 做雙向多尺度融合，最後 **Detect 頭**
（Conv + DWConv + DFL softmax）在三個尺度上輸出框與類別。
