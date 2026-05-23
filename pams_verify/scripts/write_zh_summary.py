#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures_zh"
ZH_REPORT = REPORTS / "results_and_figures_summary_zh.md"

COLORS = {
    "blue": "#2f6fbb",
    "green": "#2f9e62",
    "orange": "#e28a2d",
    "red": "#d64545",
    "purple": "#7a5cc7",
    "gray": "#6b7280",
    "light_gray": "#e5e7eb",
    "dark": "#111827",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


F_TITLE = font(36, True)
F_SUBTITLE = font(25, True)
F_LABEL = font(21)
F_SMALL = font(17)
F_TINY = font(14)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = COLORS["dark"],
    width_chars: int = 34,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=width_chars, break_long_words=False, replace_whitespace=False))
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line or " ", fnt)[1] + line_gap
    return y


def new_canvas(title: str, subtitle: str | None = None, size: tuple[int, int] = (1400, 900)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size[0], 92), fill="#f3f6fb")
    draw.text((44, 28), title, font=F_TITLE, fill=COLORS["dark"])
    if subtitle:
        draw.text((46, 98), subtitle, font=F_LABEL, fill=COLORS["gray"])
    return img, draw


def nice_max(values: list[float]) -> float:
    m = max(values) if values else 1.0
    if m <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(m))
    for step in (1, 2, 5, 10):
        if m <= step * exp:
            return step * exp
    return 10 * exp


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    title: str,
    color: str,
    value_fmt: str = "{:.2f}",
    y_max: float | None = None,
) -> None:
    x0, y0, x1, y1 = area
    draw.text((x0, y0 - 38), title, font=F_SUBTITLE, fill=COLORS["dark"])
    draw.line((x0, y1, x1, y1), fill=COLORS["gray"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["gray"], width=2)
    y_top = y_max or nice_max(values)
    for i in range(5):
        frac = i / 4
        y = y1 - int(frac * (y1 - y0))
        draw.line((x0 - 5, y, x1, y), fill="#eef0f4", width=1)
        draw.text((x0 - 62, y - 10), value_fmt.format(y_top * frac), font=F_TINY, fill=COLORS["gray"])
    bar_w = max(24, int((x1 - x0) / max(1, len(values)) * 0.52))
    gap = (x1 - x0) / max(1, len(values))
    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = x0 + gap * (idx + 0.5)
        h = 0 if y_top == 0 else int((value / y_top) * (y1 - y0))
        bx0 = int(cx - bar_w / 2)
        bx1 = int(cx + bar_w / 2)
        by0 = y1 - h
        draw.rectangle((bx0, by0, bx1, y1), fill=color)
        draw.text((bx0, by0 - 26), value_fmt.format(value), font=F_SMALL, fill=COLORS["dark"])
        tw, _ = text_size(draw, label, F_SMALL)
        draw.text((int(cx - tw / 2), y1 + 12), label, font=F_SMALL, fill=COLORS["dark"])


def draw_grouped_bar_panel(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_max: float = 1.0,
    value_fmt: str = "{:.2f}",
) -> None:
    x0, y0, x1, y1 = area
    draw.text((x0, y0 - 38), title, font=F_SUBTITLE, fill=COLORS["dark"])
    draw.line((x0, y1, x1, y1), fill=COLORS["gray"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["gray"], width=2)
    for i in range(5):
        frac = i / 4
        y = y1 - int(frac * (y1 - y0))
        draw.line((x0 - 5, y, x1, y), fill="#eef0f4", width=1)
        draw.text((x0 - 56, y - 10), value_fmt.format(y_max * frac), font=F_TINY, fill=COLORS["gray"])
    group_w = (x1 - x0) / max(1, len(labels))
    bar_w = max(16, int(group_w * 0.22))
    for li, label in enumerate(labels):
        cx = x0 + group_w * (li + 0.5)
        offsets = [-(len(series) - 1) * bar_w * 0.65 + si * bar_w * 1.3 for si in range(len(series))]
        for offset, (_, values, color) in zip(offsets, series):
            value = values[li]
            h = int((value / y_max) * (y1 - y0))
            bx0 = int(cx + offset - bar_w / 2)
            bx1 = int(cx + offset + bar_w / 2)
            draw.rectangle((bx0, y1 - h, bx1, y1), fill=color)
        tw, _ = text_size(draw, label, F_TINY)
        draw.text((int(cx - tw / 2), y1 + 12), label, font=F_TINY, fill=COLORS["dark"])
    lx = x1 - 300
    ly = y0 - 36
    for name, _, color in series:
        draw.rectangle((lx, ly + 6, lx + 20, ly + 22), fill=color)
        draw.text((lx + 28, ly), name, font=F_SMALL, fill=COLORS["dark"])
        ly += 30


def draw_scatter(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    rows: list[dict[str, float | str]],
    title: str,
) -> None:
    x0, y0, x1, y1 = area
    draw.text((x0, y0 - 38), title, font=F_SUBTITLE, fill=COLORS["dark"])
    draw.line((x0, y1, x1, y1), fill=COLORS["gray"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["gray"], width=2)
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    x_max = nice_max(xs)
    y_max = 1.0
    for i in range(5):
        frac = i / 4
        gy = y1 - int(frac * (y1 - y0))
        gx = x0 + int(frac * (x1 - x0))
        draw.line((x0, gy, x1, gy), fill="#eef0f4", width=1)
        draw.line((gx, y0, gx, y1), fill="#f6f7f9", width=1)
        draw.text((x0 - 55, gy - 10), f"{y_max * frac:.2f}", font=F_TINY, fill=COLORS["gray"])
        draw.text((gx - 14, y1 + 10), f"{x_max * frac:.0f}", font=F_TINY, fill=COLORS["gray"])
    draw.text((x0 + 240, y1 + 45), "平均加载/union blocks", font=F_SMALL, fill=COLORS["gray"])
    draw.text((x0 - 8, y0 - 64), "决策匹配率", font=F_SMALL, fill=COLORS["gray"])
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        fa = float(row["false_accept"])
        px = x0 + int((x / x_max) * (x1 - x0))
        py = y1 - int((y / y_max) * (y1 - y0))
        color = COLORS["green"] if fa < 0.1 else COLORS["orange"] if fa < 0.2 else COLORS["red"]
        radius = 8 + int(min(18, fa * 45))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="white", width=2)
        draw.text((px + radius + 5, py - 10), str(row["label"]), font=F_TINY, fill=COLORS["dark"])


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_baseline_figure(live: dict[str, Any]) -> None:
    runs = live.get("runs", [])
    labels = [r["method"].replace("dense_no_spec", "dense").replace("ngram_", "ngram-") for r in runs]
    output = [r["benchmark_json"]["output_throughput"] for r in runs]
    itl = [r["benchmark_json"]["mean_itl_ms"] for r in runs]
    img, draw = new_canvas("Live vLLM smoke baseline", "真实 vLLM smoke：dense 在短随机 workload 上最快")
    draw_bar_panel(draw, (105, 210, 650, 735), labels, output, "Output throughput (tok/s)", COLORS["blue"])
    draw_bar_panel(draw, (805, 210, 1310, 735), labels, itl, "Mean ITL (ms, 越低越好)", COLORS["orange"])
    save(img, FIGURES / "baseline_live_smoke_zh.png")


def make_union_figure(union: dict[str, Any]) -> None:
    selected = ["independent_topk", "shared_only", "shared_fixed_residual", "pams", "oracle_shared_residual"]
    rows = [r for r in union.get("results", []) if r["method"] in selected]
    labels = [r["method"].replace("shared_fixed_residual", "fixed").replace("oracle_shared_residual", "oracle").replace("independent_topk", "indep") for r in rows]
    blocks = [r["average_union_blocks"] for r in rows]
    growth = [r["union_growth_ratio"] for r in rows]
    img, draw = new_canvas("Union growth and loaded blocks", "独立 mask 的 union 明显膨胀；PAMS 降低加载块数但还不解决正确性")
    draw_bar_panel(draw, (95, 210, 655, 735), labels, blocks, "Avg union blocks", COLORS["purple"])
    draw_bar_panel(draw, (805, 210, 1310, 735), labels, growth, "Union growth ratio", COLORS["green"], y_max=2.0)
    save(img, FIGURES / "union_and_blocks_zh.png")


def make_tradeoff_figure(mask: dict[str, Any]) -> None:
    rows = []
    for r in mask.get("results", []):
        if r["method"] in {"dense_all_blocks", "independent_topk", "shared_only", "shared_fixed_residual", "pams", "pams_fallback"}:
            rows.append(
                {
                    "label": r["method"].replace("dense_all_blocks", "dense").replace("independent_topk", "indep").replace("shared_fixed_residual", "fixed"),
                    "x": r["average_union_blocks"],
                    "y": r["decision_match_rate"],
                    "false_accept": r["false_accept_rate"],
                }
            )
    img, draw = new_canvas("Quality / memory tradeoff", "横轴越小越省 KV；纵轴越高越正确；点越红 false accept 越高")
    draw_scatter(draw, (110, 220, 1260, 740), rows, "Loaded blocks vs decision match")
    draw_wrapped(
        draw,
        (900, 122),
        "趋势：PAMS 更省块，但 sparse 决策的 false accept 仍高，因此不能作为可提交 token 的近似 verifier。",
        F_SMALL,
        fill=COLORS["red"],
        width_chars=32,
    )
    save(img, FIGURES / "quality_risk_tradeoff_zh.png")


def make_false_accept_figure(correctness: dict[str, Any], mask: dict[str, Any], integ_c: dict[str, Any]) -> None:
    rows_by_method = {r["method"]: r for r in mask.get("results", [])}
    labels = ["approx", "pams_fb", "C_approx", "exact"]
    fa = [
        correctness["approximate_sparse"]["false_accept_rate"],
        rows_by_method["pams_fallback"]["false_accept_rate"],
        integ_c["approximate_fallback"]["false_accept_rate"],
        correctness["exact_fallback"]["false_accept_rate"],
    ]
    fr = [
        correctness["approximate_sparse"]["false_reject_rate"],
        rows_by_method["pams_fallback"]["false_reject_rate"],
        integ_c["approximate_fallback"]["false_reject_rate"],
        correctness["exact_fallback"]["false_reject_rate"],
    ]
    img, draw = new_canvas("False accept / false reject", "0.1% GO 阈值远低于当前 approximate 路径")
    draw_grouped_bar_panel(
        draw,
        (110, 215, 1260, 735),
        labels,
        [("false accept", fa, COLORS["red"]), ("false reject", fr, COLORS["orange"])],
        "错误率对比",
        y_max=0.25,
    )
    draw.line((110, 733 - int((0.001 / 0.25) * 520), 1260, 733 - int((0.001 / 0.25) * 520)), fill=COLORS["green"], width=2)
    draw.text((980, 702), "GO false accept threshold = 0.1%", font=F_SMALL, fill=COLORS["green"])
    save(img, FIGURES / "false_accept_summary_zh.png")


def make_ablation_figure(ablation: dict[str, Any]) -> None:
    rows = ablation.get("results", [])
    labels = [r["ablation"].replace("no_", "no ").replace("dense_fallback_all_early", "all dense") for r in rows]
    labels = [label[:12] for label in labels]
    decision = [r["decision_match_rate"] for r in rows]
    false_accept = [r["false_accept_rate"] for r in rows]
    img, draw = new_canvas("Ablation summary", "当前最有效的正确性手段是 dense fallback，而不是单独的 prior/reach/risk")
    draw_grouped_bar_panel(
        draw,
        (105, 220, 1290, 735),
        labels,
        [("decision match", decision, COLORS["blue"]), ("false accept", false_accept, COLORS["red"])],
        "Ablation: match 越高越好，false accept 越低越好",
        y_max=1.0,
    )
    save(img, FIGURES / "ablation_summary_zh.png")


def make_dashboard() -> None:
    img, draw = new_canvas("PAMS-Verify GO / NO-GO dashboard", "最终结论：NO-GO，但 offline motivation 成立")
    cards = [
        ("完成", "14 个实验目录完成注册；offline 分析、correctness、ablation、live smoke baseline 均有结果。", COLORS["green"]),
        ("最强正结果", "PAMS 将 avg union blocks 从 14.01 降到 7.93；block efficiency proxy 提升到 0.401。", COLORS["blue"]),
        ("关键失败", "没有 patched-vLLM sparse verifier end-to-end；Integration B 未编译/未运行。", COLORS["red"]),
        ("正确性风险", "approx sparse false accept = 14.13%，远高于 0.1% GO 阈值。", COLORS["orange"]),
        ("真实 baseline", "dense smoke 84.62 tok/s；ngram-4/8 约 81.3 tok/s，短随机 workload 下没有收益。", COLORS["purple"]),
        ("下一步", "使用 editable vLLM checkout，先做 exact scheduler hook，再做 attention backend mask prototype。", COLORS["gray"]),
    ]
    x_positions = [70, 500, 930]
    y_positions = [190, 500]
    for idx, (title, body, color) in enumerate(cards):
        x = x_positions[idx % 3]
        y = y_positions[idx // 3]
        draw.rounded_rectangle((x, y, x + 380, y + 250), radius=16, fill="#f9fafb", outline=color, width=4)
        draw.text((x + 24, y + 24), title, font=F_SUBTITLE, fill=color)
        draw_wrapped(draw, (x + 24, y + 76), body, F_LABEL, width_chars=19)
    draw.rectangle((70, 800, 1330, 850), fill="#fff1f2", outline=COLORS["red"], width=2)
    draw.text((92, 812), "判定：NO-GO。不能把 offline proxy 或 CPU reference microbench 写成 end-to-end speedup。", font=F_LABEL, fill=COLORS["red"])
    save(img, FIGURES / "go_nogo_dashboard_zh.png")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(data: dict[str, Any]) -> None:
    live = data["live"].get("runs", [])
    live_rows = "\n".join(
        f"| `{r['method']}` | {r['benchmark_json']['request_throughput']:.3f} | {r['benchmark_json']['output_throughput']:.3f} | {r['benchmark_json']['mean_ttft_ms']:.3f} | {r['benchmark_json']['mean_itl_ms']:.3f} | {r['benchmark_json']['mean_e2el_ms']:.3f} |"
        for r in live
    )
    union_rows = "\n".join(
        f"| `{r['method']}` | {r['average_union_blocks']:.3f} | {r['union_growth_ratio']:.3f} | {r['decision_match_rate']:.3f} | {pct(r['false_accept_rate'])} | {r['accepted_tokens_per_loaded_block']:.3f} |"
        for r in data["union"].get("results", [])
    )
    mask_rows = "\n".join(
        f"| `{r['method']}` | {r['average_union_blocks']:.3f} | {r['decision_match_rate']:.3f} | {pct(r['false_accept_rate'])} | {pct(r['false_reject_rate'])} | {pct(r['dense_fallback_rate'])} | {r['accepted_tokens_per_loaded_block']:.3f} |"
        for r in data["mask"].get("results", [])
    )
    ablation_rows = "\n".join(
        f"| `{r['ablation']}` | {r['avg_loaded_blocks']:.3f} | {r['decision_match_rate']:.3f} | {pct(r['false_accept_rate'])} | {pct(r['false_reject_rate'])} | {pct(r['dense_fallback_rate'])} |"
        for r in data["ablation"].get("results", [])
    )
    hardware = data["hardware"]
    memory = data["memory"]
    acceptance = data["acceptance"]
    correctness = data["correctness"]
    integ_b = data["integration_b"]

    content = f"""# PAMS-Verify 中文结果汇总与趋势结论

## 1. 一页式结论

最终判定：**NO-GO**。

这轮工作已经完成了 `pams_verify/` 实验框架、硬件/显存预检、synthetic trace、offline union/mask/correctness/ablation 分析、CPU reference sparse-kernel microbenchmark，以及真实 vLLM dense/ngram smoke baseline。

但是最终系统结论不成立：当前没有真实 patched-vLLM PAMS sparse verifier end-to-end 结果。vLLM 0.20.0 在 Qwen3-8B 上走 `FLASH_ATTN`，该路径只有常规 paged KV `block_table` / sequence length 元数据，没有 per-request arbitrary verifier block-mask API。因此 Integration B 没有编译、没有运行，也没有 end-to-end speedup。

![GO/NO-GO dashboard](figures_zh/go_nogo_dashboard_zh.png)

**核心趋势：**

- 独立 sparse mask 会造成明显 union 膨胀：`independent_topk` 的 union growth ratio 是 `1.863x`。
- PAMS 能显著减少 loaded blocks：avg union blocks 从 independent 的 `14.008` 降到 `7.926`。
- PAMS 的 block efficiency proxy 更好：accepted tokens / loaded block 达到 `0.401`。
- 但 sparse verifier 正确性不过关：approximate sparse false accept 是 `{pct(correctness['approximate_sparse']['false_accept_rate'])}`，远高于 `0.1%` GO 阈值。
- 真实 vLLM smoke baseline 中，dense 比 ngram-4/8 略快；没有任何 PAMS end-to-end 加速结果。

## 2. 环境与显存约束

| 项目 | 数值 |
|---|---:|
| GPU | {hardware.get('gpu_name')} |
| VRAM | {hardware.get('total_vram_gb')} GB |
| Driver | {hardware.get('driver_version')} |
| CUDA | {hardware.get('cuda_version')} |
| PyTorch | {hardware.get('torch_version')} |
| vLLM | {hardware.get('vllm_version')} |
| Target model | Qwen/Qwen3-8B |
| 推荐 dtype | {hardware.get('recommended_dtype')} |
| 估算权重显存 | {memory.get('model_weight_memory_gb')} GB |
| KV bytes/token | {memory.get('kv_bytes_per_token')} |
| 推荐 max_model_len | {memory.get('recommended_max_model_len')} |

趋势和结论：

- RTX 5090 32GB 对 Qwen3-8B BF16 是可跑但紧张的配置。
- 8192/16384 不能默认放开，需要按 estimator 降低 `max_num_seqs` 或先跑 smoke。
- 本轮真实 vLLM smoke 使用更保守的 `max_model_len=2048`、`max_num_seqs=4`。

## 3. Live vLLM smoke baseline

| 方法 | req/s | output tok/s | TTFT ms | ITL ms | E2E ms |
|---|---:|---:|---:|---:|---:|
{live_rows}

![Live baseline](figures_zh/baseline_live_smoke_zh.png)

趋势和结论：

- dense_no_spec: `84.619 tok/s`，mean ITL `10.578 ms`。
- ngram-4/8: 都约 `81.3 tok/s`，mean ITL 约 `11.0 ms`。
- 在这个短随机 smoke workload 上，ngram speculative 没有带来收益，反而略慢。
- 这只是 baseline smoke，不是完整 end-to-end matrix。

## 4. Union problem：动机是否成立

| 方法 | avg union blocks | union growth | decision match | false accept | accepted tokens / loaded block |
|---|---:|---:|---:|---:|---:|
{union_rows}

![Union and blocks](figures_zh/union_and_blocks_zh.png)

原始实验图：

![Union growth vs draft length](../experiments/03_union_problem/figures/union_growth_vs_draft_len.png)

![Jaccard overlap](../experiments/03_union_problem/figures/jaccard_overlap.png)

![Coverage vs union blocks](../experiments/03_union_problem/figures/coverage_vs_union_blocks.png)

趋势和结论：

- `independent_topk` 每个 token 平均只看 `7.121` 个 blocks，但 speculative block 的 union 达到 `14.008`，说明独立 mask 的复用性差。
- `shared_only` 的 union 最小，但 recall 和 correctness 更差。
- `PAMS` 在 loaded blocks 和 proxy efficiency 上更好，但 target attention recall 和 false accept 仍然不安全。
- 结论：**PAMS 的问题动机成立，但正确性还没有解决。**

## 5. Acceptance prior：是否有可用信号

| Split | ECE | Brier | AUROC accept | Temperature |
|---|---:|---:|---:|---:|
| calibration | {acceptance['calibration']['ece']:.4f} | {acceptance['calibration']['brier']:.4f} | {acceptance['calibration']['auroc_accept']:.4f} | {acceptance['calibration']['temperature']:.4f} |
| validation | {acceptance['validation']['ece']:.4f} | {acceptance['validation']['brier']:.4f} | {acceptance['validation']['auroc_accept']:.4f} | {acceptance['validation']['temperature']:.4f} |
| test | {acceptance['test']['ece']:.4f} | {acceptance['test']['brier']:.4f} | {acceptance['test']['auroc_accept']:.4f} | {acceptance['test']['temperature']:.4f} |

![Calibration curve](../experiments/04_acceptance_prior/figures/calibration_curve.png)

![Reach probability vs usefulness](../experiments/04_acceptance_prior/figures/reach_probability_vs_usefulness.png)

![Wasted blocks by depth](../experiments/04_acceptance_prior/figures/wasted_blocks_by_depth.png)

![Risk vs fallback](../experiments/04_acceptance_prior/figures/risk_vs_fallback.png)

趋势和结论：

- test AUROC accept 是 `{acceptance['test']['auroc_accept']:.4f}`，说明 draft-side prior 有一定预测能力。
- `rho` 对 useful token 的 AUROC 是 `{acceptance['test']['auroc_useful_from_rho']:.4f}`，比单纯 acceptance prior 更贴近“哪些 token 值得分配 verifier budget”。
- 但这只是 offline trace 信号，不能直接证明 end-to-end 加速。

## 6. Offline mask planner：省 block 与正确性的 tradeoff

| 方法 | avg union blocks | decision match | false accept | false reject | dense fallback | accepted tokens / loaded block |
|---|---:|---:|---:|---:|---:|---:|
{mask_rows}

![Quality risk tradeoff](figures_zh/quality_risk_tradeoff_zh.png)

![Pareto quality vs loaded blocks](../experiments/05_mask_planner_offline/figures/pareto_quality_vs_loaded_blocks.png)

![False accept reject](../experiments/05_mask_planner_offline/figures/false_accept_reject.png)

![Accepted tokens per loaded block](../experiments/05_mask_planner_offline/figures/accepted_tokens_per_loaded_block.png)

![Fallback tradeoff](../experiments/05_mask_planner_offline/figures/fallback_tradeoff.png)

趋势和结论：

- `PAMS` 的 avg union blocks 是 `7.926`，比 `independent_topk` 的 `14.008` 低很多。
- `PAMS` 的 accepted tokens / loaded block 是 `0.401`，proxy 上优于 dense 和 independent。
- 但 `PAMS` false accept 是 `21.43%`；`pams_fallback` 也仍有 `16.43%` false accept。
- 结论：**offline memory proxy 有改善，但 verifier correctness 不可接受。**

## 7. Correctness：最终失败的关键

| 模式 | decision match | false accept | false reject | dense fallback | greedy token-ID exact |
|---|---:|---:|---:|---:|---|
| approximate_sparse | {correctness['approximate_sparse']['decision_match_rate']:.3f} | {pct(correctness['approximate_sparse']['false_accept_rate'])} | {pct(correctness['approximate_sparse']['false_reject_rate'])} | {pct(correctness['approximate_sparse']['dense_fallback_rate'])} | false |
| exact_fallback | {correctness['exact_fallback']['decision_match_rate']:.3f} | {pct(correctness['exact_fallback']['false_accept_rate'])} | {pct(correctness['exact_fallback']['false_reject_rate'])} | {pct(correctness['exact_fallback']['dense_fallback_rate'])} | true |

![False accept summary](figures_zh/false_accept_summary_zh.png)

![False accept false reject](../experiments/11_correctness_quality/figures/false_accept_false_reject.png)

趋势和结论：

- approximate sparse 的 false accept 是 `{pct(correctness['approximate_sparse']['false_accept_rate'])}`，这是系统论文 claim 的硬失败点。
- exact fallback 能做到 token-ID exact，但 dense fallback rate 是 `100%`，因此没有 sparse verifier 节省。
- 当前不能声称 approximate sparse verifier 是质量安全的。

## 8. Sparse kernel microbenchmark

| 方法 | mean latency ms |
|---|---:|
| dense | {data['kernel']['mean_latency_ms_by_method']['dense']:.3f} |
| shared_only | {data['kernel']['mean_latency_ms_by_method']['shared_only']:.3f} |
| pams | {data['kernel']['mean_latency_ms_by_method']['pams']:.3f} |
| shared_fixed_residual | {data['kernel']['mean_latency_ms_by_method']['shared_fixed_residual']:.3f} |
| independent_sparse | {data['kernel']['mean_latency_ms_by_method']['independent_sparse']:.3f} |

![Kernel latency vs union blocks](../experiments/06_sparse_kernel_microbench/figures/kernel_latency_vs_union_blocks.png)

趋势和结论：

- 本轮是 CPU reference path，不是 GPU Triton kernel。
- sparse reference 比 dense reference 更慢，说明当前 microbench 只能说明实现开销，不能作为 GPU 加速证据。
- 自定义 Triton sparse attention kernel 未实现。

## 9. vLLM integration：为什么没跑通 PAMS end-to-end

| Integration | patched vLLM | compiled | live vLLM | outcome |
|---|---:|---:|---:|---|
| A scheduler hook | false | false | false | offline policy only |
| B attention sparse verifier | false | false | false | unsupported backend, no patch applied |
| C fallback prefilter | false | false | false | offline simulation only |

Integration B 关键事实：

- vLLM package: `{integ_b['vllm_inspection']['package_path']}`
- vLLM version: `{integ_b['vllm_inspection']['version']}`
- PAMS feature flag present: `{integ_b['pams_feature_flag_present']}`
- arbitrary block mask supported: `{integ_b['arbitrary_block_mask_supported']}`
- limitation: `{integ_b['limitation']}`

趋势和结论：

- A/C 都只是 offline policy，不满足 final GO。
- B 是最关键路径，但当前 installed vLLM 后端没有 arbitrary verifier mask 接口。
- 因此 end-to-end matrix 中所有 PAMS 方法都被标记为 `blocked_no_patched_vllm_sparse_verifier`。

## 10. Ablation：哪些因素真正有效

| ablation | avg loaded blocks | decision match | false accept | false reject | dense fallback |
|---|---:|---:|---:|---:|---:|
{ablation_rows}

![Ablation summary zh](figures_zh/ablation_summary_zh.png)

![Ablation summary original](../experiments/12_ablations/figures/ablation_summary.png)

趋势和结论：

- `dense_fallback_all_early` 正确性最好，但本质上回到了 dense verification。
- `no_fallback` false accept 仍有 `21.43%`。
- 去掉 acceptance prior / reach probability / risk term 后，结果没有出现决定性崩溃，说明当前 synthetic setup 中这些项还不是强证据。
- 结论：**fallback 是当前正确性的主导因素，PAMS 的 prior/reach/risk 还没有证明为 end-to-end 必要组件。**

## 11. 最终判断

这轮结果最合理的表述是：

> PAMS-Verify 目前是一个 offline research prototype。它证明了 independent per-token sparse masks 会导致 KV block union growth，也显示 shared/residual planning 能改善 accepted-token-per-loaded-block 这个 proxy。但它还没有证明真实 vLLM end-to-end speedup，也没有证明 approximate sparse verifier 的正确性安全。

不能声称：

- PAMS 已经实现 end-to-end 加速。
- 当前 sparse verifier 可以安全提交 token。
- CPU reference microbenchmark 代表 GPU kernel speedup。
- offline loaded-block proxy 等价于系统吞吐提升。

下一步建议：

1. 使用 editable vLLM source checkout。
2. 先实现 Integration A 的 exact scheduler hook，建立一个可跑通的 exact live baseline。
3. 再实现 attention backend mask prototype，让 verifier attention 能接收 per-request PAMS block mask。
4. 最后重新跑 long_rag_4k/8k end-to-end，并以 false accept `<0.1%` 作为硬门槛。
"""
    ZH_REPORT.write_text(content, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = {
        "hardware": load_json(EXPERIMENTS / "00_env" / "hardware.json"),
        "memory": load_json(EXPERIMENTS / "00_env" / "memory_estimate.json"),
        "live": load_json(EXPERIMENTS / "01_dense_baselines" / "parsed" / "live_vllm_baseline_smoke.json"),
        "union": load_json(EXPERIMENTS / "03_union_problem" / "parsed" / "union_metrics.json"),
        "acceptance": load_json(EXPERIMENTS / "04_acceptance_prior" / "parsed" / "acceptance_prior_metrics.json"),
        "mask": load_json(EXPERIMENTS / "05_mask_planner_offline" / "parsed" / "mask_planner_metrics.json"),
        "kernel": load_json(EXPERIMENTS / "06_sparse_kernel_microbench" / "parsed" / "kernel_summary.json"),
        "integration_b": load_json(EXPERIMENTS / "08_vllm_integration_b_attention_patch" / "parsed" / "integration_b_result.json"),
        "integration_c": load_json(EXPERIMENTS / "09_vllm_integration_c_fallback_prefilter" / "parsed" / "integration_c_result.json"),
        "correctness": load_json(EXPERIMENTS / "11_correctness_quality" / "parsed" / "correctness_metrics.json"),
        "ablation": load_json(EXPERIMENTS / "12_ablations" / "parsed" / "ablation_metrics.json"),
    }
    make_dashboard()
    make_baseline_figure(data["live"])
    make_union_figure(data["union"])
    make_tradeoff_figure(data["mask"])
    make_false_accept_figure(data["correctness"], data["mask"], data["integration_c"])
    make_ablation_figure(data["ablation"])
    write_report(data)
    print(f"Wrote {ZH_REPORT}")
    print(f"Wrote figures under {FIGURES}")


if __name__ == "__main__":
    main()
