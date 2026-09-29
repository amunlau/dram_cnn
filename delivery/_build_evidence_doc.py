# -*- coding: utf-8 -*-
"""Rebuild evidence Word: more code shots + longer captions (6–12 pages)."""
from __future__ import annotations

from pathlib import Path
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = Path("/root/workspace")
OUT_DIR = ROOT / "delivery"
SHOT_DIR = OUT_DIR / "基线模型剪枝与量化代码完成证明_截图"
SHOT_DIR.mkdir(parents=True, exist_ok=True)
DOC_PATH = OUT_DIR / "基线模型剪枝与量化代码完成证明.docx"
FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"


def set_run_font(run, size=11, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Times New Roman"
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), "Noto Serif CJK SC")
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")


def add_para(doc, text, *, size=11, bold=False, space_after=8, first_line=True, center=False):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.3
    if first_line and not center:
        pf.first_line_indent = Cm(0.74)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)
    return p


def add_h(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run(text)
    set_run_font(r, size=15 if level == 1 else 12.5, bold=True)
    return p


def extract(path, a, b):
    lines = Path(path).read_text(encoding="utf-8").splitlines()[a - 1 : b]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def render(code, out, title):
    fmt = ImageFormatter(
        style="default",
        line_numbers=True,
        font_name="DejaVu Sans Mono",
        font_size=11,
        line_number_chars=4,
        line_number_bg="#f0f0f0",
        line_number_fg="#666666",
        image_pad=8,
        line_pad=2,
    )
    img = Image.open(BytesIO(highlight(code, PythonLexer(), fmt))).convert("RGB")
    font = ImageFont.truetype(FONT_CJK, 14)
    bh = 28
    w = max(img.width + 2, 620)
    h = img.height + bh + 2
    canvas = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, w, bh], fill="#2c3e50")
    d.text((8, 6), title, fill="#ecf0f1", font=font)
    canvas.paste(img, (max(0, (w - img.width) // 2), bh))
    d.rectangle([0, 0, w - 1, h - 1], outline="#bdc3c7")
    canvas.save(out, format="PNG", optimize=True)
    return out


def add_shot(doc, img_path, caption):
    im = Image.open(img_path)
    aspect = im.height / max(im.width, 1)
    # Keep pages in ~6–12 range with many shots: slightly smaller figures
    w = 5.0
    if aspect > 1.2:
        w = 4.5
    if aspect > 1.45:
        w = 4.0
    if aspect > 1.7:
        w = 3.6
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.first_line_indent = Cm(0)
    p.add_run().add_picture(str(img_path), width=Inches(w))
    cp = doc.add_paragraph()
    cp.paragraph_format.space_before = Pt(1)
    cp.paragraph_format.space_after = Pt(8)
    cp.paragraph_format.line_spacing = 1.25
    cp.paragraph_format.first_line_indent = Cm(0.74)
    r = cp.add_run(caption)
    set_run_font(r, size=10.5)


SNIPPETS = [
    (
        "01",
        "src/dram2cnn/pipelines/run_prune.py",
        43,
        90,
        "pipelines/run_prune.py（参数定义）",
        "本段展示剪枝入口的参数解析逻辑。用户可指定基线训练会话目录、检查点路径、剪枝方法列表（BN-γ / Filter / 非结构化 L1·L2）以及稀疏或通道剪除比例档位；"
        "同时可配置最少保留通道数、是否跨层全局排序、是否保护 stem/末层，以及剪枝后短时微调的 epoch 数。"
        "这些开关直接对应基线模型上「多方法、多比例」的 FP32 剪枝对比实验设定。",
    ),
    (
        "02",
        "src/dram2cnn/engine/prune/bn_channel.py",
        1,
        45,
        "engine/prune/bn_channel.py（BN-γ 剪枝接口）",
        "本模块实现基于批归一化尺度参数的结构化通道剪枝。函数 _bn_gamma_scores 读取各 Conv–BN–ReLU 块中 BatchNorm2d.weight（即 γ）的绝对值，作为通道重要性分数；"
        "prune_bn_channels 再按目标剪除比例选择保留通道，并调用通用拓扑裁剪例程。该实现对应 Network Slimming 思想中的「用 γ 选通道」阶段，"
        "用于在基线模型上验证通道级冗余是否可被安全削减。",
    ),
    (
        "03",
        "src/dram2cnn/engine/prune/bn_channel.py",
        50,
        85,
        "engine/prune/bn_channel.py（剪枝执行与统计）",
        "本段是 BN-γ 剪枝的执行主体：先记录剪前模型统计，再按层保护策略生成 keep 索引，调用 apply_channel_keeps 完成实际张量切片，"
        "随后用 verify_forward 校验剪后网络仍可前向。返回字典中包含方法名、目标比例、通道前后数量、参数与理论算力变化等，"
        "便于与其他剪枝方法在同一口径下汇总对比。",
    ),
    (
        "04",
        "src/dram2cnn/engine/prune/filter_prune.py",
        1,
        55,
        "engine/prune/filter_prune.py（Filter L1）",
        "本模块实现 Filter L1 结构化剪枝。_filter_l1_scores 对每个输出滤波器计算权重绝对值之和（L1 范数）作为重要性；"
        "prune_filters_l1 的参数接口与 BN-γ 方法对齐（比例、最少保留通道、保护层等），但评分完全来自卷积核幅度，不依赖 BN 是否存在或 γ 是否稀疏化。"
        "因此它是基线模型实验中与 BN-γ 配对的硬件友好结构化对照。",
    ),
    (
        "05",
        "src/dram2cnn/engine/prune/unstructured.py",
        13,
        70,
        "engine/prune/unstructured.py（非结构化 L1 与掩码固化）",
        "本模块实现非结构化权重稀疏。_prunable_params 收集全部 Conv2d/Linear 权重；prune_unstructured_l1 使用 torch.nn.utils.prune 的全局 L1 剪枝，"
        "按目标稀疏度将小幅度权重量零。finalize_unstructured_pruning 在微调结束后把掩码固化进权重，避免「名义稀疏但落盘已变稠密」的错误结论。"
        "同文件另有非结构化 L2 实现，以 w² 作为 importance_scores；对标量权重而言 |w| 与 w² 诱导的排序等价，故实验中 L1/L2 结果应一致——"
        "代码路径已同时覆盖 L1 与 L2，本截图仅展示 L1 与固化流程。该路径用于估计在拓扑不变条件下基线模型仍可承受的权重级冗余。",
    ),
    (
        "06",
        "src/dram2cnn/engine/prune/structured_utils.py",
        47,
        90,
        "engine/prune/structured_utils.py（保留通道选择）",
        "本函数根据每层通道分数选择保留索引。支持逐层同比例与跨层全局排序两种模式，并可强制保护指定层（例如 stem / 末层）；"
        "同时保证每层至少保留 min_keep_channels 个通道，避免过度剪空。该逻辑被 BN-γ 与 Filter L1 共用，"
        "使两种结构化方法在「宽度削减规则」一致的前提下，仅因重要性分数不同而保留不同通道集合。",
    ),
    (
        "07",
        "src/dram2cnn/engine/prune/structured_utils.py",
        112,
        165,
        "engine/prune/structured_utils.py（跨层同步裁剪）",
        "apply_channel_keeps 是结构化剪枝的核心拓扑操作：按 keep 索引切片当前层卷积输出权重与偏置、同步裁剪 BN，"
        "并将下一层卷积输入通道与上一层输出 keep 对齐；最后更新分类头第一层全连接的输入维与模型 out_channels。"
        "缺少任一步都会导致形状不匹配。该模块保证基线模型在通道剪枝后仍可正确前向与继续微调。",
    ),
    (
        "08",
        "src/dram2cnn/engine/prune/finetune.py",
        1,
        45,
        "engine/prune/finetune.py（剪枝后短时微调）",
        "short_finetune 在训练集上对剪后模型做短时交叉熵微调，用于恢复结构化剪枝后的精度下降。"
        "注释明确：若非结构化剪枝尚未 finalize，前向会自动乘 weight_mask，微调不会破坏有效稀疏；调用方应在结束后再固化。"
        "该模块是基线模型剪枝对比实验中「剪后参数适配」的统一实现。",
    ),
    (
        "09",
        "src/dram2cnn/engine/prune/runner.py",
        49,
        105,
        "engine/prune/runner.py（对比编排入口）",
        "run_prune_compare 是剪枝对比的总编排函数。它加载基线模型权重，构建验证集与测试集 DataLoader，"
        "并支持跳过已有 baseline 评估以便多卡分片并行。输出目录下会写入各方法、各比例的指标与统计，"
        "从而把分散的剪枝算法组织成可复现的基线模型 FP32 剪枝对比实验。",
    ),
    (
        "10",
        "src/dram2cnn/engine/quantize_official.py",
        54,
        110,
        "engine/quantize_official.py（对称 QConfig 与可量化模型）",
        "本段包含对称有符号 INT8 的 QConfig 定义，以及 QuantizableGridCNN 的结构封装。"
        "make_symmetric_qconfig 将权重与激活均设为 per-tensor 对称 qint8（零点为 0）；"
        "QuantizableGridCNN 把原 GridCNN 的 Conv–BN–ReLU 展开为 Sequential，并加入 QuantStub/DeQuantStub，"
        "同时预留 stem 与分类末层保持 FP32 的保护策略。前向路径为 stem → 量化区间 stages/池化/全连接 → dequant → head_out（末层 logit 仍为 FP32）。"
        "这是后续 BN 折叠与 FakeQuant 的基础载体。",
    ),
    (
        "11",
        "src/dram2cnn/engine/quantize_official.py",
        163,
        221,
        "engine/quantize_official.py（BN 折叠与 FakeQuant 插入）",
        "collect_fuse_pairs 枚举可折叠的 Conv–BN–ReLU 路径；fold_batchnorm 在 eval 下调用 fuse_modules 完成 BN 折叠；"
        "prepare_fakequant 在确认无残留 BatchNorm2d 后插入对称 FakeQuant。该顺序是官方量化路径的关键正确性约束："
        "必须先得到部署友好的无 BN 拓扑，再做 FakeQuant，否则评估图与打包权重会与真实部署不一致。",
    ),
    (
        "12",
        "src/dram2cnn/engine/quantize_official.py",
        232,
        288,
        "engine/quantize_official.py（校准与 QDQ 导出）",
        "calibrate 在训练集上前向若干 batch，令 observer 统计激活动态范围并确定 scale，对应训练后量化（PTQ）的校准阶段；"
        "export_qdq_onnx 导出含 QuantizeLinear/DequantizeLinear 的 ONNX，并强制校验 BatchNormalization 节点数为 0、Q/DQ 节点存在。"
        "这两段共同证明：量化不仅完成了仿真，也产出了可被后续推理引擎摄取、且拓扑无 BN 的交付图。",
    ),
    (
        "13",
        "src/dram2cnn/engine/quantize_official.py",
        335,
        385,
        "engine/quantize_official.py（INT8 打包与审计）",
        "pack_fakequant_int8 遍历带 weight_fake_quant 的模块，按对称尺度将折叠后权重量化为 int8 并写入打包文件；"
        "stem 与 head_out 作为保护层保留 FP32。打包后通过 audit_pack_vs_fakequant 检查重构误差，"
        "确保整数权重与 FakeQuant 仿真对象一致。该模块把「可评估的 FakeQuant 模型」落实为可交付的 INT8 权重产物。",
    ),
    (
        "14",
        "src/dram2cnn/engine/quantize_official.py",
        423,
        474,
        "engine/quantize_official.py（准备模型与 QAT 训练）",
        "build_prepared_model 将「加载基线检查点 → BN 折叠 → 插入 FakeQuant」封装为一步，供 PTQ/QAT 共用；"
        "run_qat_epochs 在折叠后的 FakeQuant 图上继续短训，训练期保持 observer 与 fake quant 开启，使权重适应量化噪声。"
        "这对应官方路径中的量化感知训练阶段，与仅校准的 PTQ 形成对照。",
    ),
    (
        "15",
        "src/dram2cnn/pipelines/run_quantize_official.py",
        1,
        55,
        "pipelines/run_quantize_official.py（量化实验入口）",
        "本文件是官方对称 INT8 实验的命令行入口。它解析基线训练会话与检查点，支持单卡或 torchrun 分布式启动，"
        "并将具体算法委托给 run_official_quant。产物目录命名体现 PTQ+QAT 流程，便于与剪枝实验目录区分。"
        "该入口证明量化工作已具备可重复执行的工程化启动方式。",
    ),
    (
        "16",
        "src/dram2cnn/engine/quantize_official.py",
        614,
        670,
        "engine/quantize_official.py（run_official_quant：FP32 段）",
        "run_official_quant 是官方量化主流程。截图所示为流程开头：创建输出目录、记录 API/拓扑/保护层等元信息，"
        "并先评估折叠前的 FP32 基线指标与检查点体积，作为后续 PTQ/QAT 对比的参照。"
        "统一评价仍采用验证集重搜阈值后再在测试集报告的口径。",
    ),
    (
        "17",
        "src/dram2cnn/engine/quantize_official.py",
        684,
        730,
        "engine/quantize_official.py（run_official_quant：PTQ/QAT 段）",
        "本段展示 PTQ 执行链：build_prepared_model 完成 BN 折叠与 FakeQuant，calibrate 校准后 freeze，"
        "再评估、导出 QDQ ONNX、打包 INT8，并记录 pack 审计误差；随后进入 QAT 短训分支。"
        "至此，基线模型上的「BN 折叠 → PTQ → QAT → 导出/打包」完整代码路径均已覆盖，可用于证明量化研究与实现已经完成。",
    ),
]


def main() -> None:
    # clean old shots
    for p in SHOT_DIR.glob("*.png"):
        p.unlink()

    height_sum = 0.0
    items = []
    for sid, rel, a, b, title, cap in SNIPPETS:
        if "Base-new" in cap or "base-new" in cap.lower():
            raise RuntimeError("caption contains Base-new")
        code = extract(ROOT / rel, a, b)
        out = SHOT_DIR / f"{sid}.png"
        banner = f"{title}  (L{a}–{b})"
        render(code, out, banner)
        im = Image.open(out)
        aspect = im.height / im.width
        w = 5.0
        if aspect > 1.2:
            w = 4.5
        if aspect > 1.45:
            w = 4.0
        if aspect > 1.7:
            w = 3.6
        height_sum += w * aspect + 0.95
        items.append((out, cap))
        print(f"{sid}: {im.size}")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(1.6)
    sec.bottom_margin = Cm(1.6)

    add_para(doc, "基线模型剪枝与量化代码完成证明", size=18, bold=True, center=True, first_line=False, space_after=4)
    add_para(
        doc,
        "（FP32 剪枝对比 + 官方对称 INT8：BN 折叠 → PTQ / QAT）",
        size=11,
        center=True,
        first_line=False,
        space_after=12,
    )
    add_para(
        doc,
        "本文档用于证明：针对 GridCNN-Deep 内存故障预测任务的基线模型，项目侧已完成两条相对独立的后处理实现——"
        "其一为 FP32 剪枝方法对比（BN-γ 通道剪枝、Filter L1 结构化剪枝、非结构化 L1/L2 幅度稀疏，以及统一编排与入口）；"
        "其二为官方对称 INT8 量化（先批归一化折叠，再以 torch.ao.quantization Eager FakeQuant 开展 PTQ 与 QAT，"
        "并导出无 BN 的 QDQ ONNX、打包 INT8 权重）。正文以关键源码截图为主，每张截图配一段文字，说明所属模块及其功能作用。"
        "全文统一使用「基线模型」表述。",
        space_after=10,
    )

    add_h(doc, "一、工作范围概述")
    add_para(
        doc,
        "剪枝路径以基线模型检查点为输入，在多种方法与多档比例上完成剪枝、可选短时微调与验证集重搜阈值评估，"
        "用于证明结构化与非结构化压缩代码均已落地。量化路径不叠加剪枝，专注位宽压缩与部署拓扑一致性，"
        "用于证明 BN 折叠、对称 FakeQuant、PTQ/QAT、导出与打包链路完整可用。",
        space_after=10,
    )

    add_h(doc, "二、FP32 剪枝对比相关代码")
    for img, cap in items[:9]:
        add_shot(doc, img, cap)

    add_h(doc, "三、官方对称 INT8（BN 折叠 → PTQ / QAT）相关代码")
    for img, cap in items[9:]:
        add_shot(doc, img, cap)

    add_h(doc, "四、小结")
    add_para(
        doc,
        "以上共计十七段关键代码截图，分别覆盖剪枝入口与参数、BN-γ / Filter L1 / 非结构化 L1·L2、通道选择与跨层同步裁剪、"
        "短时微调与对比编排，以及对称 INT8 配置、可量化封装、BN 折叠、FakeQuant、校准、QDQ 导出、INT8 打包、QAT 训练与官方主流程。"
        "据此可证明：面向基线模型的剪枝与量化研究工作，在代码实现层面已经完成，并具备可复现的实验入口与产物输出。",
        space_after=6,
    )
    add_para(
        doc,
        "主要代码目录：src/dram2cnn/pipelines/run_prune.py、src/dram2cnn/engine/prune/；"
        "src/dram2cnn/pipelines/run_quantize_official.py、src/dram2cnn/engine/quantize_official.py。",
        size=10,
        first_line=False,
        space_after=2,
    )

    doc.save(DOC_PATH)
    text = "\n".join(p.text for p in Document(DOC_PATH).paragraphs)
    assert "Base-new" not in text and "base-new" not in text.lower()
    print("Wrote", DOC_PATH)
    print(f"Size {DOC_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"n_shots={len(items)} approx_pages≈{round((height_sum + 3) / 9.5, 1)}")
    print("OK")


if __name__ == "__main__":
    main()
