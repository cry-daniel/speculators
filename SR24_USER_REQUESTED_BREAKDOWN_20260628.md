# SR24 User-Requested Seven-Part Breakdown, 2026-06-28

## Breakdown Pivot, 2026-06-28

本轮先停止普通 sweep，按用户要求把慢点拆成 scheduler/mask、base sparse
Linear、residual correction、gather/scatter、routing、CUDA Graph 和 GPU util
七类。最新聚合报告在：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_slowdown_pivot_breakdown_20260628_v2/report.md
```

交叉参考的 current-candidate 报告在：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_candidate_breakdown_bs64_math256_20260628_combined/seven_part_report/report.md
```

读法必须分开：

- `clean_serving` 行只用于端到端 tok/s、GPU util、CUDA Graph、接受长度。
- `diagnostic_component_timing` 行只用于定位 Linear 和 routing 成本；它会插入
  CUDA event / sync，tok/s 和 GPU util 不能当最终性能。

当前最重要的结论：

| part | evidence | read |
| --- | --- | --- |
| clean throughput | row-routed `speclink_t08` 在 bs64/math256 约 `1.09-1.11x` dense full-batch tok/s；total tok/s 约 `1.04-1.10x` | 有收益但还没有到 `1.2x` |
| GPU util | clean `speclink_t08` 约 `93-94%`，peak `100%` | GPU 不是 idle；慢点不是简单 underutilization |
| base sparse Linear | diagnostic `gate_up_proj=16-31` 约 `1.08-1.12ms/call`，base sparse aggregate 约 `0.96-1.01ms/call` | 当前最大的局部 GPU-side 成本 |
| residual correction | row-routed dense-row correction 约 `0.16-0.18ms/call`；force-all Triton bucket correction 约 `0.18-0.20ms/call`；all-corrected compressed correction 约 `0.57-0.69ms/call` | mixed row-routed correction本身不是最大项，但它叠加在 sparse base 后；all-corrected correction 太重 |
| gather/scatter | row-routed 约 `0.016-0.017ms/call`，all-corrected 约 `0.03-0.135ms/call` | 当前不是第一瓶颈 |
| routing | row-routed diagnostic draft residual fraction 约 `0.53-0.54`，non-draft residual fraction 约 `0.40-0.46`，bucket fill 约 `0.98-0.99` | residual/protected rows 仍太多，导致 mixed path 做了不少重复 work |
| CUDA Graph | clean 行可以保持健康；diagnostic eager 行全是 `NONE` | 不要用 diagnostic 行判断真实 graph 性能 |

因此现在的慢点不是接受长度塌了、不是 GPU 空转，也不是单纯的
`index_select/index_add_` 开销。关键问题是 mixed useful-work shape：为了保护一批
residual rows，当前实现仍然先对大部分 verify rows 做一次 sparse base，然后再对
protected/dense rows 做 correction。这个 two-pass 成本把 `base_only_24` 的上界收益
吃掉了。后续优化应该优先减少 residual rows 或改成 graph-safe fused/packed mixed
operator；只做 CPU 侧 route rows、bucket size 小调、或 gather/scatter-only cleanup
优先级较低。

## Latest Refresh

最新可复用报告在：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_refresh_20260628/report.md
```

这份报告重新归并了 clean serving、clean runtime stats、instrumented serving
和 CPU-sync ablation，并修正了两个展示问题：负的 CUDA Graph delta 会被过滤，
同一 method/dataset/batch 的 clean runtime graph 统计会补到 clean serving
代表行。因此 clean `speclink_t08` 现在直接显示 `{"FULL": 78, "NONE": 2}`，
而不是误以为 graph 未测。

当前慢点结论不变：clean `speclink_t08` 的 GPU util 约 `93.7%`，CUDA Graph
`FULL` 占比约 `97.5%`，scheduler/mask clean 路径是 sub-ms 级；慢主要不是
GPU 空转、graph miss 或接受长度崩掉，而是 mixed path 做了低效的重复 GPU
work：先对大量 rows 做 sparse base Linear，再对一批 residual/protected rows
做 dense correction。下一轮优化应该把这个 seven-part breakdown 当作固定
gate，不能只看端到端 tok/s。

质量侧需要单独处理。`critical_prefix` 把 draft residual prefix 从 `4` 提到
`8` 后，GSM8K-50 paired regression 从 `6` 个降到 `3` 个，但 accuracy 仍从
`0.7800` 掉到 `0.7200`。这说明当前质量掉点不只是 early/full-residual
fastpath 缺失，而是 selective routing 在一部分后续 reasoning token 上仍然不够
保守。对应 artifact：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_minprefix8_quality_upper_gsm8k50_20260628/report.md
```

## 2026-06-28 Route-All / Direct-CPU Follow-Up

质量根因已经被单独定位：`FULL_DECODE_ONLY` 这类强行动态图配置会让 selective
SR24 在 GSM8K 上出现 paired regression，即使把实现退化成 all-residual
densefastpath no-op 也会掉点；切回默认 vLLM compile 后，no-op 和实际 selective
`speclink_t08` 都能在 GSM8K-50 上恢复到 dense 对齐。

质量通过的关键 artifact：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_allresidual_nohook_defaultcompile_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_selective_defaultcompile_quality_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_defaultcompile_quality_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_directcpu_defaultcompile_quality_gsm8k50_20260628/report.md
```

其中 direct-CPU route rows 质量也通过：dense 和 `speclink_t08` 都是 `0.7800`，
paired regression 为 `0`。但它不是性能优化路径。bs64/math256 的 route-all
吞吐对比如下：

| variant | dense full tok/s | speclink full tok/s | full speedup | dense total tok/s | speclink total tok/s | total speedup | key scheduler time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| route-all, default compile | `3437.167` | `3352.785` | `0.975x` | `2590.114` | `2504.687` | `0.967x` | `row_indices=32.436ms/step` |
| route-all + direct CPU rows, default compile | `3496.863` | `3343.234` | `0.956x` | `2589.078` | `2497.103` | `0.965x` | `direct_cpu_rows=36.662ms/step` |

Direct-CPU rows 只把原来的 GPU `nonzero()/row_indices` 同步开销换成了 Python
`list/set` 构造和 CPU->GPU row-index 拷贝。它证明了当前 route-all mixed path
确实有 row-index/routing 开销，但也证明这条路不能在 CPU 侧修。后续要么：

1. 不再显式构造 full residual/base row list，而是让 GPU mask 直接驱动 fused
   或 packed mixed operator；
2. 写一个 GPU compact/route kernel，在生成 residual mask 的同一侧产出
   residual/base rows；
3. 更根本地降低 residual rows，避免 route-all 变成接近全 dense correction。

对应 throughput artifact：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_defaultcompile_throughput_bs64_math256_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_directcpu_defaultcompile_throughput_bs64_math256_20260628/seven_part_report/report.md
```

## 2026-06-28 Row-Routed Follow-Up

当前更有希望的方向是 row-routed MLP / gate-up path，而不是 route-all。它把
“所有 row 先 sparse base，再对 residual row dense 覆盖”的 two-pass 形式，改成
对 dense/base row 分开算一部分 MLP，再只组装最终 hidden-size 输出。这个方向已经
能把 clean `speclink_t08` 从 route-all 的 `<1.0x` 推到大约 `1.1x`，但仍未达到
`1.2x`。

| variant | dense full tok/s | speclink full tok/s | full speedup | dense total tok/s | speclink total tok/s | total speedup | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| route-all default compile | `3437.167` | `3352.785` | `0.975x` | `2590.114` | `2504.687` | `0.967x` | quality-safe but slower than dense |
| row-routed, bucket=16 | `3173.157` | `3477.607` | `1.096x` | `2779.114` | `2944.526` | `1.060x` | current best clean row-routed candidate |
| row-routed, bucket=64 | `3115.661` | `3444.199` | `1.105x` | `2611.152` | `2742.896` | `1.050x` | larger bucket does not clearly improve total throughput |
| row-routed, bucket=8 | `3197.482` | `3454.284` | `1.080x` | `2754.086` | `2824.464` | `1.026x` | smaller bucket is worse; bucket cap is not the main missing 1.2x lever |

The bucket=8 run also exposed and fixed a runner issue: SR24 preset application
was overwriting explicit CLI overrides for `--sr24-residual-bucket-size`. The
matrix runner now preserves explicit SR24 bucket/row-routed/graph override flags
after applying presets, so future bucket sweeps are meaningful.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_k8_clean_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_gateup_bucket64_k8_clean_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_bucket8_overridefixed_k8_clean_bs64_math256_20260628_goal/report.md
```

Current interpretation: row-routed removes the worst route-all row-index
overhead and keeps CUDA Graph coverage (`profile:{"FULL":49}` in the clean
rows), but the dense branch still handles too many rows and the sparse/base
branch shape is not efficient enough to reach `1.2x`. The next useful
optimization should change residual scope/controller or implement a better
packed/fused MLP operator; simply shrinking the bucket from 16 to 8 is not a
solution.

## Current Chinese Read

这次应该先停下普通参数 sweep，按七项 breakdown 定位慢点。当前最可信的读法是：
clean serving 行只看端到端吞吐、GPU util、CUDA Graph 和接受长度；diagnostic 行只看
CUDA event/operator attribution，不拿它的 tok/s 做性能结论。

当前主要依据：

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_pivot_bs64_math_k8_20260628/seven_part_report_with_runtime/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_speclink_t08_bs64_math_k8_20260628/seven_part_report/report.md
```

核心结论：现在慢不是因为接受长度塌了，也不是 GPU 没跑满。`base_only_24`
在 clean serving 中能到约 `1.196x` dense full-batch tok/s，GPU util 约
`94%`，说明结构化稀疏本身有上界收益。`speclink_t08` 只有约 `1.099x`
full-batch tok/s，原因是 mixed path 的有用计算效率差：先对很多行算 sparse
base Linear，再对大量 residual/protected rows 做 dense correction，相当于重复
做了一部分 GPU work。

| 部分 | 当前测到什么 | 结论 |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` 约 `0.378-0.400ms/step`；row bucket 约 `0.110-0.130ms/step`；sync-heavy 诊断可到十几到几十 ms/step | clean 路径不是主瓶颈；只要不引入 `.item()`/CPU 同步，mask/bucket 构造目前是次要项 |
| base sparse linear | diagnostic aggregate sparse base 约 `0.986ms/call`；`gate_up_proj=16-31` 约 `1.101ms/call` | 当前最大的局部 GPU-side 成本 |
| residual correction | dense-row correction 约 `0.164ms/call`；bucket rows/call 通常 `16` | 单独看比 sparse base 小，但它是叠加在 sparse base 后面的重复工作 |
| gather/scatter | 约 `0.016ms/call` | 不是第一瓶颈；只优化 scatter 很难带来 1.2x |
| routing 统计 | draft residual/base 约 `12712/12712`，non-draft residual/base 约 `3178/3801`，bucket fill 约 `0.976` | residual/protected rows 太多，导致 mixed path 经常接近“两遍算” |
| CUDA Graph | clean 行可到 `{"FULL":78,"NONE":2}`；CPU-sync ablation 中低同步行仍高 util，sync-heavy 会掉到全 NONE | clean graph coverage 可以健康；不要用 sync-heavy 诊断行判断真实 graph 性能 |
| GPU util | clean `speclink_t08` 约 `93.7%`，peak `100%` | GPU 很忙；问题是忙在低效/重复的 useful work，不是 underutilization |

CPU 同步消融补充说明：`sync_heavy` 会把 total tok/s 打到约 `1558`，GPU util
降到约 `61.7%`，说明 CPU/GPU 同步确实会严重伤性能；但当前 clean low-sync
路径在 `stats on/off` 间只小幅波动，说明当前主瓶颈已经不是普通 CPU 统计开销。

下一步优化优先级：

1. 优先减少 two-pass mixed operator 的重复计算：不要对最终会 dense-overwrite 的
   rows 继续完整算 sparse base，或者做 graph-safe fused/packed mixed kernel。
2. 优先找质量安全的 controller，让 residual rows 明显下降；必须配 paired
   accuracy gate，不能只看 aggregate accuracy。
3. CUDA Graph 只作为 guardrail：新候选如果 `NONE` 比例高就先修 graph，但当前
   clean path 不是主要 graph 问题。
4. 不要再优先做普通 threshold/bucket sweep、gather/scatter-only rewrite、
   或 all-residual fallback。已有结果显示这些最多只能修边角，不能解释或解决
   `speclink_t08` 没达到 `1.2x` 的根因。

This note records the focused breakdown requested before another optimization
sweep. It separates clean serving rows from instrumented rows: clean rows are
the throughput/GPU-util/CUDA-Graph reference, while instrumented rows localize
Linear and routing costs but add synchronization overhead.

Primary artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_seven_part_breakdown_bs64_math256_20260628/seven_part_report/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency 64,
max new tokens 256 for clean serving, SR24 scoped to `gate_up_proj=16-31`,
`high_confidence` threshold `0.3`, `dense_rows@cuda`, residual bucket size 12,
bucket priority enabled, bonus priority 0, draft-position priority scale 1,
bucket dense copy, static mask/bucket buffers, and CUDA Graph bucket capture.
This is a focused slowdown read, not the broader bucket16/direct-cuSPARSELt
best-candidate row.

## Clean Serving

| method | total tok/s | full-batch tok/s | speedup vs dense full | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2312.012` | `3420.653` | `1.000x` | `1.697` | `88.429%` | n/a |
| `base_only_24` | `2790.222` | `3956.379` | `1.157x` | `2.027` | `90.833%` | `{"FULL":126,"NONE":2}` |
| `all_corrected_24` | `2321.099` | `3432.622` | `1.003x` | `1.697` | `88.214%` | n/a |
| `speclink_t08` | `2375.998` | `3558.964` | `1.040x` | `1.898` | `90.571%` | `{"FULL":126,"NONE":2}` |

Clean read: the GPU is not idle and CUDA Graph coverage is healthy. `base_only`
has real headroom, but `speclink_t08` recovers only a small fraction of that
headroom once residual correction is added.

## Seven-Part Diagnosis

| part | measured value | read |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` wall `0.584ms/step`; row bucket `0.085ms/step`; bucket build `0.083ms/step`; mixed row indices `0.002ms/step` | sub-ms in the clean path; not the main bottleneck |
| base sparse Linear | diagnostic `gate_up_proj=16-31` sparse base `0.994ms/call`, about `248` rows/call | largest localized GPU-side cost |
| residual correction | diagnostic dense-row correction `0.177ms/call`, bucket rows/call `12` | secondary per call, but paid on top of sparse base |
| gather/scatter | diagnostic `0.014ms/call` | not the first bottleneck in this row |
| routing statistics | diagnostic draft residual/base `8702/4682`, non-draft residual/base `1673/1824`, draft residual fraction `0.650`, non-draft residual fraction `0.478`, bucket fill `0.976` | too many rows still need residual protection |
| CUDA Graph | clean `speclink_t08` `{"FULL":126,"NONE":2}` | graph misses are not the current cause |
| GPU util | clean `speclink_t08` avg `90.571%`, peak `100%` | slowdown is inefficient useful work, not underutilization |

## Operator Read

Component microbench confirms the same pattern. For gate/up
`512/28672/4096`, sparse base alone is about `0.65x` dense under CUDA Graph,
but the current mixed proxy becomes `1.03x` dense at residual fraction `0.125`
and `1.53x` dense at residual fraction `0.5`. For down `512/4096/14336`, low
residual fractions can still beat dense, but it also loses once residual
fraction reaches `0.5`.

## Conclusion

The current slowdown is not primarily accepted-length collapse, CPU mask-build
overhead, CUDA Graph loss, or idle GPU. The slow part is the mixed useful-work
shape: the system pays sparse base Linear for many rows and then pays dense
residual correction for a large protected subset. The next optimization should
either reduce residual rows with a paired accuracy gate, or replace the current
two-pass sparse-base plus dense-row correction with a graph-safe fused/packed
GPU operator. Scheduler-sync or gather/scatter-only cleanup is lower priority
unless a later breakdown moves those numbers.

## Follow-Up

The later bucket16/direct-cuSPARSELt candidate passed a compile-aligned
GSM8K-50 paired accuracy gate, but it still did not reach the `1.2x` throughput
target. With default vLLM compile it was speed-neutral (`0.986x` total,
`0.997x` full-batch), and with SR24 `FULL_DECODE_ONLY` compile it reached only
`1.047x` total and `1.104x` full-batch.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_directcslt_bucket16_quality_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket16_directcslt_compilealigned_throughput_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket16_directcslt_srcompile_throughput_bs64_math256_20260628/report.md
```

So the current split is clear: the latest guarded candidate can preserve
GSM8K-50 behavior, but the speed bottleneck remains operator-side useful work,
not the scheduler/mask path.

## Follow-Up Ablations

I also checked two likely alternatives before changing direction:

| variant | quality read | total tok/s speedup | full-batch speedup | SR24 GPU util | graph read |
| --- | --- | ---: | ---: | ---: | --- |
| VLLM_COMPILE large graph | GSM8K-30 pair reg `0`, pair imp `2` | `0.942x` | `1.014x` | `83.63%` | `{"FULL":49,"PIECEWISE":76}` |
| Triton bucket dense correction | GSM8K-30 exact paired match, reg/imp `0/0` | `0.980x` | `0.991x` | `92.75%` | `{"FULL":44,"PIECEWISE":44}` |
| raw FULL_DECODE_ONLY reference | fastest observed, but quality-unsafe in stricter gates | `1.047x` | `1.104x` | `93.91%` | `{"FULL":49}` |

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_vllmcompile_largegraph_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_vllmcompile_largegraph_throughput_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_triton_bucket_dense_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_triton_bucket_dense_throughput_bs64_math256_20260628/report.md
```

This narrows the next optimization target. Larger quality-safe vLLM graph
capture does not recover speed, and Triton correction does not fix throughput.
The valid path now needs either a quality-safe version of the raw full-graph
execution path, or a packed/fused mixed operator that avoids computing sparse
base output for rows that dense correction will overwrite.

## Raw Full-Graph Bucket Copy Check

The latest raw `FULL_DECODE_ONLY` check separates three bucket-copy semantics:

| variant | GSM8K-30 pair reg/imp | bs64 math total speedup | bs64 math full speedup | read |
| --- | ---: | ---: | ---: | --- |
| dense-copy all selected bucket rows | `0/0` | `1.087x` | `1.109x` | quality-clean, speed not enough |
| active-only bucket scatter | `2/1` | not used | not used | unsafe |
| Triton bucket dense GEMM | `2/1` | `1.215x` | `1.140x` | fast but unsafe |

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_quality_gsm8k30_20260628_rerun/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_throughput_bs64_math256_20260628_rerun/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_activeonly_scatter_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_activeonly_triton_quality_gsm8k30_20260628/report.md
```

So the active-only direction should be treated as a failed ablation. The
quality-preserving behavior is conservative over-correction of the selected
bucket. The remaining performance work is still operator-side: reduce the
duplicated sparse-base plus dense-correction work without losing that quality
guard.

## 2026-06-28 CPU-Sync / Score-Overhead Follow-Up

按照“先减少 CPU 侧同步”的建议，补跑了三个 clean serving 消融，全部保持
Llama-3.1-8B、`math_reasoning`、bs64、EAGLE3 K=8、max new tokens 256、
bucket16/direct-cuSPARSELt、row-routed gate-up/down 形状不变：

| variant | full-batch tok/s | same-root full speedup | total tok/s | same-root total speedup | read |
| --- | ---: | ---: | ---: | ---: | --- |
| current critical-prefix | `3477.607` | `1.096x` | `2944.526` | `1.060x` | 当前参考点 |
| uniform-direct only | `3483.572` | `1.091x` | `2908.895` | `1.043x` | 与 current 基本持平，差异在噪声范围 |
| GPU-count builder | `3284.246` | `1.013x` | `2681.776` | `0.944x` | 变慢，不应作为默认优化 |
| fixed-prefix / no score-prob | `3455.529` | `1.082x` | `2928.927` | `1.050x` | 去掉 selected-prob score 收集没有明显收益 |
| sorted bucket rows | `3478.020` | `1.096x` | `2882.888` | `1.029x` | gather/index_copy 访存顺序不是主要瓶颈 |

需要注意：同时打开 `gpu_count_mask_builder` 和 `batched_uniform_direct` 时，
代码会优先走 GPU-count 路径，uniform-direct 不会触发；所以又单独补了
uniform-direct-only。结论是 CPU staging/score 收集可以作为 guardrail，但不是
missing `1.2x` 的主瓶颈。当前主线仍应放在 mixed operator 的 useful-work
效率上：要么减少需要 dense residual protection 的 rows，要么实现
graph-safe packed/fused GPU operator，避免对之后会被 dense 覆盖的 rows 仍先算
sparse base。

随后补的 bucket-row 排序也没有收益：full-batch tok/s 与 current 基本一致，
total tok/s 反而略低。因此 gather/scatter 侧的小修不应作为主线继续投入。

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_score_overhead_ablation_20260628_v2/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_score_bucket_ablation_20260629/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_uniform_direct_only_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_low_cpu_sync_builder_counts_uniform_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix4_lowscore_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sort_bucket_rows_bs64_math256_20260628/report.md
```
