# -*- coding: utf-8 -*-
"""Generate docs/yolo11_modules.pptx — YOLO11 五大模組架構簡報."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

NAVY = "0A2540"
TEAL = "1C7293"
MINT = "2EC4B6"
DARK = "13293D"
PANEL = "EEF3F8"
TEXT = "1A2733"
MUTED = "5A6B7B"
WHITE = "FFFFFF"
ZH = "Microsoft JhengHei"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    if bg:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = RGBColor.from_string(bg)
    return s


def _ea(run, font):
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def setrun(run, text, size, color, bold=False, font=ZH, italic=False):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    run.font.name = font
    _ea(run, font)


def box(slide, x, y, w, h, lines, anchor=MSO_ANCHOR.TOP, fill=None,
        line=None, line_w=1.0, round_=False, shadow=False):
    shp_type = MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE
    if fill is not None or line is not None or round_:
        shp = slide.shapes.add_shape(shp_type, Inches(x), Inches(y), Inches(w), Inches(h))
        if round_:
            try:
                shp.adjustments[0] = 0.06
            except Exception:
                pass
        if fill is None:
            shp.fill.background()
        else:
            shp.fill.solid()
            shp.fill.fore_color.rgb = RGBColor.from_string(fill)
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = RGBColor.from_string(line)
            shp.line.width = Pt(line_w)
        shp.shadow.inherit = False
        tf = shp.text_frame
    else:
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.06)
    tf.margin_bottom = Inches(0.06)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ln.get("align", PP_ALIGN.LEFT)
        p.space_after = Pt(ln.get("sa", 4))
        p.space_before = Pt(ln.get("sb", 0))
        if ln.get("ls"):
            p.line_spacing = ln["ls"]
        for r in ln["runs"]:
            setrun(p.add_run(), r[0], r[1], r[2], r.get("b", False) if isinstance(r, dict) else (len(r) > 3 and r[3]),
                   font=(r[4] if len(r) > 4 else ZH))
    return tf


def circle(slide, x, y, d, fill, text=None, tcolor=WHITE, tsize=22, tbold=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    shp.line.fill.background()
    shp.shadow.inherit = False
    if text:
        tf = shp.text_frame
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        setrun(p.add_run(), text, tsize, tcolor, bold=tbold)
    return shp


def arrow(slide, x, y, w, h, fill=MINT):
    shp = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def pill(slide, x, y, w, h, text, fill=TEAL, tcolor=WHITE, tsize=11, font=MONO):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    try:
        shp.adjustments[0] = 0.5
    except Exception:
        pass
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.1)
    tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    setrun(p.add_run(), text, tsize, tcolor, font=font)
    return shp


def slide_title(slide, num, title_en, title_zh, loc):
    circle(slide, 0.6, 0.5, 0.8, MINT, num, NAVY, 26)
    box(slide, 1.6, 0.45, 6.6, 0.95, [
        {"runs": [(title_en, 30, NAVY, True)], "sa": 0},
        {"runs": [(title_zh, 16, TEAL, False)], "sb": 2},
    ])
    pill(slide, 8.45, 0.62, 4.25, 0.42, loc, TEAL, WHITE, 10.5)


# ============================================================ 1 Title
s = slide(NAVY)
box(s, 0.9, 2.15, 11.5, 2.4, [
    {"runs": [("YOLO11 架構解析", 52, WHITE, True)], "sa": 6},
    {"runs": [("五大模組　—　作用、數學與銜接", 24, MINT, False)], "sb": 6},
], )
box(s, 0.95, 4.55, 11.5, 0.5, [
    {"runs": [("依據 ", 13, "9FB3C8"), ("ultralytics_yolo11/cfg/models/11/yolo11.yaml", 13, "9FB3C8", False, MONO),
              ("　·　全模型 24 層（0–23）：Backbone → Neck → Head", 13, "9FB3C8")]},
])
mods = ["Conv", "C3k2", "SPPF", "C2PSA", "Detect+DFL"]
for i, m in enumerate(mods):
    cx = 0.95 + i * 2.45
    pill(s, cx, 5.5, 2.2, 0.55, m, DARK if i % 2 else TEAL, MINT if i % 2 else WHITE, 14)

# ============================================================ 2 Overview / positioning
s = slide(WHITE)
box(s, 0.6, 0.45, 12.0, 0.7, [{"runs": [("五大模組住在網路的哪裡", 32, NAVY, True)]}])
box(s, 0.62, 1.15, 12.0, 0.4, [{"runs": [
    ("輸入 640×640×3", 13, MUTED, False, MONO), ("　逐段往右傳遞，最後輸出偵測框", 13, MUTED)]}])

stages = [
    (0.6, "Backbone　主幹", "層 0–10　抽特徵", TEAL,
     ["Conv ×7　stride-2 下採樣 P1→P5", "C3k2 ×5　主力特徵抽取", "SPPF　多尺度感受野（層 9）", "C2PSA　全域注意力（層 10）"]),
    (5.0, "Neck　頸部", "層 11–22　跨尺度融合", "2C6E8F",
     ["FPN 上採樣：語意往高解析度傳", "PAN 下採樣：定位往深層傳", "Upsample + Concat + C3k2", "產出 P3 / P4 / P5 三尺度"]),
    (9.4, "Head　偵測頭", "層 23　預測", "215C77",
     ["Detect　解耦雙分支", "DFL　分佈式框回歸", "三尺度各自回歸 + 分類", "→ NMS → 最終結果"]),
]
for x, name, sub, col, items in stages:
    box(s, x, 1.75, 3.6, 0.95, [
        {"runs": [(name, 18, WHITE, True)], "sa": 1, "align": PP_ALIGN.CENTER},
        {"runs": [(sub, 12, "D6E6F2")], "align": PP_ALIGN.CENTER},
    ], fill=col, round_=True, anchor=MSO_ANCHOR.MIDDLE)
    box(s, x, 2.85, 3.6, 2.35, [
        {"runs": [("•  " + it, 13, TEXT)], "sa": 7, "ls": 1.0} for it in items
    ], fill=PANEL, round_=True)
arrow(s, 4.25, 2.05, 0.7, 0.55)
arrow(s, 8.65, 2.05, 0.7, 0.55)

box(s, 0.6, 5.5, 12.1, 0.45, [{"runs": [("解析度金字塔：下採樣換語意，不同尺度分工不同大小的物件", 15, NAVY, True)]}])
pyr = [
    ["層級", "stride", "特徵圖", "每格原圖", "負責物件"],
    ["P3", "8", "80×80", "8×8 px", "小物件"],
    ["P4", "16", "40×40", "16×16 px", "中物件"],
    ["P5", "32", "20×20", "32×32 px", "大物件 / 全圖語意"],
]
colw = [1.6, 1.6, 2.0, 2.2, 4.7]
rx = 0.6
ry = 6.0
for ci, cw in enumerate(colw):
    for ri in range(4):
        cx = rx + sum(colw[:ci])
        cy = ry + ri * 0.36
        fill = NAVY if ri == 0 else (PANEL if ri % 2 else WHITE)
        tcol = WHITE if ri == 0 else TEXT
        box(s, cx, cy, cw, 0.36, [
            {"runs": [(pyr[ri][ci], 12, tcol, ri == 0)], "align": PP_ALIGN.CENTER, "sa": 0}
        ], fill=fill, line="C9D6E2", line_w=0.5, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================ module slide builder
def module_slide(num, en, zh, loc, role_lines, math_lines):
    s = slide(WHITE)
    slide_title(s, num, en, zh, loc)
    box(s, 0.6, 1.75, 0.18, 4.9, [], fill=MINT)
    box(s, 0.95, 1.7, 5.55, 0.5, [{"runs": [("作用 — 它在做什麼、為什麼", 18, TEAL, True)]}])
    box(s, 0.95, 2.25, 5.6, 4.5, [
        {"runs": [("•  ", 14, MINT, True)] + [r], "sa": 9, "ls": 1.05} for r in role_lines
    ])
    box(s, 6.85, 1.75, 5.9, 4.9, math_lines, fill=DARK, round_=True)
    return s


module_slide(
    "1", "Conv", "卷積 + 批次正規化 + SiLU，全網最底層積木", "conv.py:41 · 層 0,1,3,5,7…",
    [("結構：Conv2d → BatchNorm → SiLU 三合一", 14, TEXT),
     ("SiLU 平滑非單調，負區仍有梯度，不易死神經元", 14, TEXT),
     ("bias=False：偏置交給 BN 的平移項 β", 14, TEXT),
     ("用 stride-2 卷積取代池化下採樣，下採樣也能學特徵", 14, TEXT),
     ("部署時 BN 可融合進卷積權重，省一次運算", 14, TEXT)],
    [{"runs": [("數學重點", 16, MINT, True)], "sa": 10},
     {"runs": [("y = SiLU(BN(W ∗ x))", 17, WHITE, False, MONO)], "sa": 8},
     {"runs": [("SiLU(z) = z·σ(z) = z / (1 + e^(−z))", 15, "BFE9E0", False, MONO)], "sa": 14},
     {"runs": [("── 推論時融合 BN ──", 13, MINT, False, MONO)], "sa": 8},
     {"runs": [("W' = W · γ / √(σ² + ε)", 15, WHITE, False, MONO)], "sa": 6},
     {"runs": [("b' = β − μγ / √(σ² + ε)", 15, WHITE, False, MONO)], "sa": 12},
     {"runs": [("→ 卷積、正規化、激活綁成一個可融合單元", 13, "9FB3C8")]}],
)

module_slide(
    "2", "C3k2", "CSP 特徵塊，YOLO11 主力（取代 v8 的 C2f）", "block.py:1074 · 層 2,4,6,8…22",
    [("輸入經 cv1 切兩半：一半直通、一半逐級過 Bottleneck", 14, TEXT),
     ("所有中間輸出全部串接，再用 cv2 融合", 14, TEXT),
     ("CSP 只算一半通道 → 省約一半算力", 14, TEXT),
     ("每個中間輸出直連 cv2，多條梯度路徑（類 DenseNet）", 14, TEXT),
     ("c3k 開關：淺層用單一 Bottleneck，深層換巢狀 C3k", 14, TEXT)],
    [{"runs": [("數學重點", 16, MINT, True)], "sa": 10},
     {"runs": [("y = cv2( cat[ y0, y1, B(y1), B²(y1), … ] )", 15, WHITE, False, MONO)], "sa": 12},
     {"runs": [("Bottleneck（殘差）：", 14, "BFE9E0")], "sa": 6},
     {"runs": [("y = x + cv2(cv1(x))", 15, WHITE, False, MONO)], "sa": 6},
     {"runs": [("∂y/∂x = I + ∂F/∂x", 15, WHITE, False, MONO)], "sa": 12},
     {"runs": [("→ 永遠有一條係數為 1 的梯度高速公路", 13, "9FB3C8")], "sa": 10},
     {"runs": [("e=0.5（淺層 0.25）內部再壓通道；vs C2f 的 e=1.0", 13, "9FB3C8")]}],
)

module_slide(
    "3", "SPPF", "空間金字塔池化（快速版），補大感受野", "block.py:213 · 層 9（P5 末端）",
    [("用連續三次 5×5 MaxPool，等效取得多種感受野", 14, TEXT),
     ("四路（含原始）串接後用 1×1 融合", 14, TEXT),
     ("MaxPool 無參數、平移不變，挑出區域最強響應", 14, TEXT),
     ("放在 20×20 的 P5 末端，擴大感受野最便宜", 14, TEXT),
     ("為頸部準備好要往各尺度分發的「全域摘要」", 14, TEXT)],
    [{"runs": [("數學重點", 16, MINT, True)], "sa": 10},
     {"runs": [("max 池化可疊加：", 14, "BFE9E0")], "sa": 6},
     {"runs": [("5×5 → 9×9 → 13×13", 17, WHITE, False, MONO)], "sa": 12},
     {"runs": [("四路感受野 = { 1, 5, 9, 13 }", 16, WHITE, False, MONO)], "sa": 12},
     {"runs": [("≡ SPP(k=5,9,13)", 15, WHITE, False, MONO)], "sa": 6},
     {"runs": [("但只重複算 5×5，省掉大視窗池化", 13, "9FB3C8")]}],
)

module_slide(
    "4", "C2PSA", "位置敏感注意力，YOLO11 新增的全域建模", "block.py:1576 · 層 10",
    [("Transformer 配方：Attention + FFN + 殘差", 14, TEXT),
     ("用卷積/BN 取代 LayerNorm，包進 CSP 殼省算力", 14, TEXT),
     ("讓特徵圖上任兩個位置可直接交換資訊（卷積只看局部）", 14, TEXT),
     ("只放最小的 P5（N=400），全域建模才買得起", 14, TEXT),
     ("成果再透過頸部傳播到其他尺度", 14, TEXT)],
    [{"runs": [("數學重點", 16, MINT, True)], "sa": 10},
     {"runs": [("A(i,j) = softmax_j( q_i·k_j / √d_k )", 15, WHITE, False, MONO)], "sa": 12},
     {"runs": [("out = V·A^T + DWConv_3×3(V)", 15, WHITE, False, MONO)], "sa": 4},
     {"runs": [("                       ↑ 位置編碼", 13, MINT, False, MONO)], "sa": 12},
     {"runs": [("key_dim = head_dim × 0.5", 15, WHITE, False, MONO)], "sa": 6},
     {"runs": [("→ Q/K 維度減半省成本；÷√d_k 防 softmax 飽和", 13, "9FB3C8")], "sa": 10},
     {"runs": [("注意力成本 O(N²)：放 P3(N=6400) 太貴", 13, "9FB3C8")]}],
)

module_slide(
    "5", "Detect + DFL", "anchor-free 解耦偵測頭 + 分佈式框回歸", "head.py:26 · block.py:63 · 層 23",
    [("每尺度兩條分支：cv2 回歸框、cv3 分類", 14, TEXT),
     ("anchor-free：每格中心當點，直接預測到四邊距離", 14, TEXT),
     ("分類分支用 DWConv 深度可分離卷積 → 參數更少", 14, TEXT),
     ("DFL：把每邊距離當 16-bin 機率分佈，取期望值", 14, TEXT),
     ("分佈能表達邊界不確定性，比 L1 單點回歸更穩", 14, TEXT)],
    [{"runs": [("數學重點", 16, MINT, True)], "sa": 10},
     {"runs": [("距離取期望：E[d] = Σ i · p_i ,  i=0..15", 15, WHITE, False, MONO)], "sa": 4},
     {"runs": [("固定權重 = arange(16)，不訓練", 13, "9FB3C8")], "sa": 12},
     {"runs": [("框：dist2bbox(E[d]) × stride", 15, WHITE, False, MONO)], "sa": 10},
     {"runs": [("類別：sigmoid（多標籤、無背景類）", 15, WHITE, False, MONO)], "sa": 12},
     {"runs": [("尺度分工　stride 8 / 16 / 32", 14, "BFE9E0", False, MONO)], "sa": 4},
     {"runs": [("→ 小 / 中 / 大物件", 13, "9FB3C8")]}],
)

# ============================================================ 8 銜接
s = slide(WHITE)
box(s, 0.6, 0.45, 12.0, 0.7, [{"runs": [("五大模組如何互相搭配銜接", 32, NAVY, True)]}])

flow = [
    ("① Backbone 抽特徵", TEAL,
     "Conv 逐層 stride-2 下採樣；C3k2 在每個尺度抽特徵；\nSPPF 在 P5 補大感受野；C2PSA 補一次全域上下文。"),
    ("② Neck 雙向融合", "2C6E8F",
     "FPN 上採樣把深層「語意」往高解析度傳；\nPAN 下採樣把淺層「定位」往深層傳；\n交會處用 Concat + C3k2 真正融合。"),
    ("③ Head 三尺度預測", "215C77",
     "P3 / P4 / P5 各自送進 Detect；\ncv2+DFL 解碼框、cv3+sigmoid 分類；\n最後彙整做 NMS。"),
]
fy = 1.5
for i, (name, col, desc) in enumerate(flow):
    y = fy + i * 1.55
    circle(s, 0.6, y + 0.18, 0.6, col, str(i + 1), WHITE, 22)
    box(s, 1.45, y, 11.2, 1.35, [
        {"runs": [(name, 19, col if col != "2C6E8F" else TEAL, True)], "sa": 4},
    ] + [
        {"runs": [(line, 14, TEXT)], "sa": 2, "ls": 1.0} for line in desc.split("\n")
    ], fill=PANEL, round_=True)

box(s, 0.6, 6.35, 12.1, 0.95, [
    {"runs": [("核心一句話：", 15, NAVY, True),
              ("下採樣換取語意與感受野、犧牲定位；頸部再用雙向融合，把語意與定位重新補回每個尺度。", 15, TEXT)], "sa": 3},
    {"runs": [("資料流：", 13, MUTED, True),
              ("Conv/C3k2 → SPPF → C2PSA → FPN↑ → PAN↓ → Detect+DFL → NMS", 13, MUTED, False, MONO)]},
], fill="E4F1EE", round_=True, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================ 9 特色 × 程式位置
s = slide(WHITE)
box(s, 0.6, 0.45, 12.2, 0.7, [{"runs": [("YOLO11 特色（對比 v8）與程式模組位置", 31, NAVY, True)]}])

tbl = [
    ["面向", "YOLOv8", "YOLO11", "程式位置"],
    ["主力特徵塊", "C2f（e=1.0）", "C3k2（e=0.5 + c3k 開關）", "block.py:1074"],
    ["全域建模", "無", "新增 C2PSA 注意力", "block.py:1576"],
    ["分類分支", "標準 Conv", "DWConv 深度可分離", "head.py:100"],
    ["框回歸", "DFL", "DFL（沿用，配 anchor-free）", "block.py:63"],
    ["基礎卷積", "Conv（BN+SiLU）", "Conv（相同）", "conv.py:41"],
]
colw = [2.3, 3.0, 4.0, 2.85]
tx, ty = 0.6, 1.45
rh = 0.62
for ri, row in enumerate(tbl):
    for ci, cell in enumerate(row):
        cx = tx + sum(colw[:ci])
        cy = ty + ri * rh
        if ri == 0:
            fill, tcol, bold = NAVY, WHITE, True
        else:
            fill = PANEL if ri % 2 else WHITE
            tcol, bold = TEXT, False
        mono = (ci == 3 and ri > 0)
        box(s, cx, cy, colw[ci], rh, [
            {"runs": [(cell, 12.5 if not mono else 12, tcol, bold or (ci == 2 and ri > 0), MONO if mono else ZH)],
             "align": PP_ALIGN.CENTER if ci != 0 else PP_ALIGN.LEFT, "sa": 0}
        ], fill=fill, line="C9D6E2", line_w=0.5, anchor=MSO_ANCHOR.MIDDLE)

box(s, 0.6, 5.35, 12.1, 1.75, [
    {"runs": [("三大差異一句話總結", 17, MINT if False else TEAL, True)], "sa": 8},
    {"runs": [("•  ", 14, MINT, True), ("C2f → C3k2：", 14, NAVY, True),
              ("更深、更省算力（CSP 分流 + 殘差 + c3k 控深度）", 14, TEXT)], "sa": 6},
    {"runs": [("•  ", 14, MINT, True), ("新增 C2PSA：", 14, NAVY, True),
              ("在 P5 補一次全域注意力，卷積看不到的長距關係交給它", 14, TEXT)], "sa": 6},
    {"runs": [("•  ", 14, MINT, True), ("分類用 DWConv：", 14, NAVY, True),
              ("一層 3×3 從 O(c²·9) 降到 O(c·9 + c²)，這是參數更少的主因之一", 14, TEXT)], "sa": 0},
], fill=PANEL, round_=True)

# ============================================================ 10 IoU 對照表
s = slide(WHITE)
box(s, 0.6, 0.45, 12.2, 0.7, [{"runs": [("loss 的 IoU 與「判斷正確」的 IoU 一樣嗎？", 30, NAVY, True)]}])
box(s, 0.62, 1.13, 7.6, 0.4, [{"runs": [
    ("答案：不一樣。", 14, NAVY, True),
    ("訓練用 CIoU，評估 mAP 與 NMS 用純 IoU。", 14, MUTED)]}])
pill(s, 8.55, 1.1, 1.95, 0.42, "訓練 CIoU", TEAL, WHITE, 11, ZH)
pill(s, 10.62, 1.1, 2.1, 0.42, "評估 純 IoU", NAVY, WHITE, 11, ZH)

iou_rows = [
    ("header", "用途", "哪種 IoU", "程式位置", "為什麼選這種"),
    ("ciou", "① 框回歸 loss", "CIoU", "loss.py:189", "要可微分；兩框零重疊時仍要有梯度，模型才學得動"),
    ("ciou", "② 正樣本分配", "CIoU", "tal.py:215", "當「框品質」分數，決定哪個 anchor 配哪個 GT"),
    ("plain", "③ 驗證 / mAP 判定", "純 IoU", "metrics.py:454", "要公平、標準（COCO 慣例），門檻 0.5–0.95"),
    ("plain", "④ NMS 去重", "純 IoU", "iou = 0.7", "只需衡量重疊程度，移除重複的框"),
]
colw_iou = [2.7, 1.9, 2.5, 5.0]
tx, ty = 0.6, 1.72
for ri, row in enumerate(iou_rows):
    kind = row[0]
    cells = row[1:]
    rh = 0.5 if ri == 0 else 0.72
    cy = ty if ri == 0 else ty + 0.5 + (ri - 1) * 0.72
    for ci, cell in enumerate(cells):
        cx = tx + sum(colw_iou[:ci])
        if ri == 0:
            fill, tcol, bold = NAVY, WHITE, True
        else:
            fill = "E1EFEA" if kind == "ciou" else "E9EEF5"
            tcol, bold = TEXT, False
        rcolor, rbold = tcol, bold
        if ri > 0 and ci == 1:
            rcolor = TEAL if cell == "CIoU" else NAVY
            rbold = True
        mono = (ri > 0 and ci == 2)
        box(s, cx, cy, colw_iou[ci], rh, [
            {"runs": [(cell, 11.5 if mono else 12.5, rcolor, rbold, MONO if mono else ZH)],
             "align": PP_ALIGN.LEFT if ci in (0, 3) else PP_ALIGN.CENTER, "sa": 0, "ls": 1.0}
        ], fill=fill, line="C9D6E2", line_w=0.5, anchor=MSO_ANCHOR.MIDDLE)

box(s, 0.6, 5.28, 12.1, 1.7, [
    {"runs": [("關鍵差別：為什麼不能混用", 16, TEAL, True)], "sa": 8},
    {"runs": [("純 IoU = 交集 / 聯集", 14.5, NAVY, True),
              ("　兩框沒重疊就 = 0，梯度也 = 0", 13.5, TEXT)], "sa": 7},
    {"runs": [("CIoU = IoU − 中心距離懲罰 − 長寬比懲罰", 14.5, NAVY, True),
              ("　零重疊時仍有梯度可學", 13.5, TEXT)], "sa": 7},
    {"runs": [("→ ", 13.5, MINT, True), ("loss 一定用 CIoU 家族", 13.5, NAVY, True),
              ("（要梯度）；", 13.5, TEXT), ("mAP / NMS 一定用純 IoU", 13.5, NAVY, True),
              ("（要跟論文公平比較）", 13.5, TEXT)], "sa": 0},
], fill=PANEL, round_=True)

# ============================================================ 11 設計哲學 closing
s = slide(NAVY)
box(s, 0.7, 0.55, 12.0, 0.8, [{"runs": [("設計哲學統整", 34, WHITE, True)]}])
phil = [
    ("算力花在刀口上", "高解析度層 e=0.25 瘦身、注意力只放 20×20、分類分支用 DWConv"),
    ("梯度路徑越多越好", "CSP 分流、殘差、FPN/PAN 跨層直連，全在給梯度修高速公路"),
    ("預測不確定性而非單點", "DFL 用 16-bin 分佈取代直接回歸，邊界模糊時更穩"),
    ("一份結構、五種規模", "depth / width / max_channels 三倍率複合縮放，n→x 共用同一 yaml"),
]
for i, (h, d) in enumerate(phil):
    col = i % 2
    row = i // 2
    x = 0.7 + col * 6.15
    y = 1.7 + row * 2.45
    circle(s, x, y, 0.7, MINT, str(i + 1), NAVY, 24)
    box(s, x + 0.95, y - 0.05, 5.0, 2.1, [
        {"runs": [(h, 20, MINT, True)], "sa": 6},
        {"runs": [(d, 14, "D6E6F2")], "ls": 1.1},
    ])
box(s, 0.7, 6.75, 12.0, 0.5, [{"runs": [
    ("相較 v8：", 13, "9FB3C8", True),
    ("C2f→C3k2（更深更省）、新增 C2PSA（全域注意力）、分類分支輕量化（DWConv）", 13, "9FB3C8")]}])

prs.save("docs/yolo11_modules.pptx")
print("saved docs/yolo11_modules.pptx ;", len(prs.slides._sldIdLst), "slides")
