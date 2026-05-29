我核对了 vLLM 相关接口方向：vLLM 官方文档把 speculative decoding 作为独立推理特性，scheduler / chunked prefill 相关配置也围绕 max_num_batched_tokens、max_num_seqs 等 token/sequence budget 展开；GuideLLM 可用于 serving benchmark 和 latency/throughput 测量。这个 prompt 里我让 Codex 以源码和 AGENTS.md 为准，不硬编码模型路径。 ￼

下面这版可以直接给 Codex。

你是一个资深 vLLM / speculative decoding / serving scheduler 系统工程师。你的任务是在当前 repo 中实现并验证 SpecLink-CV：confidence-guided chunked verification for EAGLE3-style linear speculative decoding。
请先阅读并遵守：
- AGENTS.md
- README / docs / scripts
- 当前 repo 中所有 speculative decoding / EAGLE3 / benchmark / GuideLLM 相关脚本
- vLLM 本地 patch 和当前安装版本源码
- examples/evaluate/eval-guidellm 下的 configs/scripts
不要硬编码模型路径。Llama3、Qwen3、对应 EAGLE3 speculator、数据集路径、运行环境、GPU 命令方式都以 AGENTS.md 和当前 repo 为准。如果 AGENTS.md 中的信息和本 prompt 冲突，以 AGENTS.md 为准，但必须在最终报告里说明差异。
本任务的核心不是做 sparse KV，而是先实现 SpecLink 的 chunked verification 路线：
DLM 先生成 K 个 draft tokens；
TLM 不一定一次性验证全部 K 个；
SpecLink-CV 先验证一个 prefix chunk；
如果 prefix 里已经 reject，则跳过 suffix verification；
如果 prefix 全 accepted，则继续验证 suffix；
所有剪枝必须基于 TLM exact verification 结果，不能用 DLM 猜测直接跳过 token。
SpecLink-CV 必须保持 speculative decoding 的 correctness。默认 greedy / temperature=0 下，SpecLink-CV 输出应与 vLLM+EAGLE3 one-shot verification 输出一致。如果不一致，必须报告 mismatch，不得把它当成成功。
========================
0. 总体目标
========================
实现并评估以下系统机制：
1. DLM confidence decides chunk size
   - 可开关。
   - 打开时，根据 DLM 对每个 draft token 的置信度估计 local acceptance probability，再选择每个 request 的 prefix chunk size。
   - 关闭时，不使用 confidence，默认 chunk size = draft token length 的一半。
   - 例如 K=8 时默认 h=4；K=12 时默认 h=6。
   - 这些开关用于后续消融实验。
2. Sync batch vs async verification queue
   - 可开关。
   - sync mode：模拟简单同步批处理策略，用于消融；可以要求同一轮 batch 内的 requests 对齐到同一种 verification chunk 策略。
   - async mode：每个 request 独立决定 chunk size，进入 verification queue；scheduler 从 queue 中挑选 ready chunks 拼成 TLM verification batch。
   - async mode 是目标设计。
3. Roofline-aware packing
   - 可开关。
   - 打开时，scheduler 在 queue 中有多个 candidate chunks 时，根据预计 FLOPs、KV bytes、batch shape、GPU utilization 和 expected pruning benefit 选择哪些 chunks 拼到一起。
   - 关闭时，使用简单 FIFO / benefit-only / bucket-only 策略。
   - 该功能用于证明：只在算法层面选择 chunk 不够，还要保证 GPU 上真的跑得高效。
最终需要对比：
1. pure vLLM
   - 无 speculative decoding。
   - 无 SpecLink。
2. vLLM + EAGLE3
   - 原始 one-shot verification。
   - 无 SpecLink-CV。
3. vLLM + EAGLE3 + SpecLink-CV ablations
   - confidence chunk sizing on/off。
   - async queue on/off。
   - roofline-aware packing on/off。
   - 共 2 x 2 x 2 = 8 组主要消融。
4. Best SpecLink-CV
   - 从消融结果中选出满足 correctness 约束下的最佳配置。
   - 最佳配置不能只看吞吐，也要看 p95/p99 latency、correctness、extra verifier pass overhead 和 skipped suffix tokens。
========================
1. 输出目录
========================
创建统一结果目录：
results/speclink_cv_TIMESTAMP/
目录结构：
00_env/
01_impl_notes/
02_unit_tests/
03_confidence_calibration/
04_baselines/
05_cv_ablation/
06_scheduler_queue/
07_roofline_packing/
08_figures/
09_reports/
logs/
patches/
scripts/
每个实验子目录必须包含：
- run_command.sh
- stdout.log
- stderr.log 或 combined.log
- raw logs/json/jsonl
- parsed csv
- config snapshot
- git commit / git diff snapshot
最终必须生成：
results/speclink_cv_TIMESTAMP/09_reports/SPECLINK_CV_REPORT.md
results/speclink_cv_TIMESTAMP/09_reports/summary_metrics.csv
results/speclink_cv_TIMESTAMP/09_reports/summary_metrics.json
不要编造结果。每个数字必须能追溯到 raw log/json/jsonl。
========================
2. 环境检查
========================
先实现并运行：
tools/speclink_cv/env_check.py
输出：
00_env/env_report.md
00_env/env_report.json
内容至少包括：
- hostname
- date
- git branch / commit / dirty diff
- conda env
- Python path
- torch version
- CUDA version
- vLLM version
- GuideLLM version
- GPU name / memory
- nvidia-smi
- torch.cuda.is_available()
- AGENTS.md 中 Llama3 / Qwen3 / EAGLE3 speculator 路径解析结果
- MTBench 数据集路径解析结果
- math_reasoning 数据集路径解析结果
- 当前 vLLM speculative decoding / EAGLE3 配置是否可用
- 当前 vLLM 是否启用 chunked prefill
- max_num_batched_tokens / max_num_seqs / scheduler config 快照
如果模型、speculator 或数据集找不到，先根据 AGENTS.md 尝试定位。失败时停止并报告，不要继续跑空实验。
========================
3. 需要新增的配置开关
========================
请在 vLLM / repo wrapper 中增加 SpecLink-CV 配置。优先使用 CLI args；如果 vLLM 当前路径中 CLI 改动太重，可以用环境变量或 JSON config，但必须文档化。
核心开关：
--speclink-cv-enable
  default: false
  是否启用 SpecLink chunked verification。
--speclink-cv-confidence-sizing
  default: false
  是否使用 DLM confidence 决定 chunk size。
  false 时，chunk size 默认为 ceil(K/2) 或 K/2；K=8 -> 4，K=12 -> 6。
--speclink-cv-async-queue
  default: false
  是否启用 async verification queue。
  false 时使用 sync batch 消融模式。
  true 时 request 独立进入 queue，由 scheduler 异步拼 batch。
--speclink-cv-roofline-packing
  default: false
  是否启用 roofline-aware packing。
其他必要参数：
--speclink-cv-candidate-chunks
  default: "1,2,4,6,8,full"
  候选 chunk size 集合。实际取 <= K 的 chunk。
  K=8 时候选为 1,2,4,8/full。
  K=12 时候选为 1,2,4,6,8,12/full。
--speclink-cv-default-half-policy
  choices: floor,ceil
  default: floor
  由于本实验 K=8/12 都是偶数，默认 half 即 K/2。
--speclink-cv-min-benefit
  default: 0.0
  confidence sizing 打开时，只有 best expected benefit > threshold 才 chunk；否则 one-shot。
--speclink-cv-max-verify-tokens-per-step
  default: inherit from vLLM max_num_batched_tokens when possible
  每轮最多调度多少 verification tokens。
--speclink-cv-max-verify-seqs-per-step
  default: inherit from max_num_seqs when possible
--speclink-cv-max-queue-wait-ms
  default: 2
  async queue 中 chunk 最长等待时间。超过则 schedule now 或 fallback one-shot，避免 p99 latency 变差。
--speclink-cv-util-threshold
  default: 0.6
  roofline-aware packing 预测 GPU utilization 低于该阈值时，不单独跑小 chunk；尝试合并、等待或 fallback。
--speclink-cv-calibration-path
  default: empty
  DLM confidence calibration 模型路径。
--speclink-cv-log-jsonl
  default: empty
  记录每个 request/chunk 的决策和结果。
--speclink-cv-profile-jsonl
  default: empty
  记录 timing 和 scheduler profiling。
--speclink-cv-debug-dump
  default: false
  用于 smoke test，不要在大实验中默认开启。
所有开关必须在日志里明确打印，方便消融表自动解析。
========================
4. DLM confidence 设计
========================
目标：用 DLM proposal-time statistics 估计每个 draft token 被 TLM 接受的概率。
对第 i 个 draft token x_i，优先从 EAGLE3 proposer / draft model logits 中提取：
- draft token id
- logprob of selected draft token: log p_i
- top1 probability
- top2 probability
- top1-top2 margin
- entropy
- draft position i
- K
- context length
- request id
- model id
- dataset id
不要用 API 返回给用户的 logprobs 代替 DLM 内部置信度，除非源码确认它们就是 proposer logits。需要从 vLLM 的 EAGLE3 proposal path 中加 instrumentation。
定义 local acceptance probability：
a_hat_i = P(x_i accepted by TLM | x_1...x_{i-1} accepted)
实现 calibration：
tools/speclink_cv/collect_confidence_traces.py
tools/speclink_cv/calibrate_acceptance.py
tools/speclink_cv/evaluate_calibration.py
校准数据来自训练/validation split：
输入特征：
- logprob
- margin
- entropy
- draft position
- K
- context length
- optionally model id / dataset id
label：
- one-shot EAGLE3 verification 中该 draft token 是否被接受，条件是前缀已被接受。
支持至少一种简单 calibration：
- logistic regression
- temperature scaling
- isotonic regression
- binning calibration
输出：
- calibration model file
- ECE
- Brier score
- reliability diagram
- per-position calibration table
如果 calibration 尚未完成，confidence sizing 可以先使用 uncalibrated proxy，但必须在日志和报告里标注。
禁止：
- 在线使用 TLM accept/reject 结果来决定当前 request 的 chunk size。
- 在 test split 上调 calibration 或阈值。
========================
5. chunk size decision
========================
当 --speclink-cv-enable=true 时，每个 request 的 EAGLE3 draft 长度为 K。
如果 --speclink-cv-confidence-sizing=false：
h = K / 2
K=8 -> h=4
K=12 -> h=6
如果 --speclink-cv-confidence-sizing=true：
枚举候选 chunk size：
H = candidate chunks <= K, including full K
对每个 h，计算 prefix survival probability：
pi_h = product_{i=1}^{h} a_hat_i
prefix reject probability：
q_h = 1 - pi_h
suffix length：
s_h = K - h
估计 suffix verification cost：
C_suffix(h) = estimated TLM cost of verifying K-h suffix tokens
估计 extra cost：
C_extra(h) 包括：
- extra TLM forward / launch
- scheduler overhead
- possible CUDA graph miss
- repeated context KV read cost
- prefix chunk GPU under-utilization
- queue waiting overhead if any
expected benefit：
Benefit(h) = q_h * C_suffix(h) - C_extra(h)
选择：
h* = argmax Benefit(h)
如果 max Benefit(h) <= --speclink-cv-min-benefit，则 h*=K，走 one-shot verification。
注意：
- DLM confidence 高时，pi_h 高，q_h 低，通常应选择 K。
- DLM confidence 低或不确定时，更可能选择小 h。
- 不能固定 first-4 作为最终方法；fixed half 只是 confidence 关闭时的消融 baseline。
日志必须记录：
- K
- candidate H
- each a_hat_i
- pi_h for each candidate
- Benefit(h)
- selected h
- reason: confidence / half / fallback / one-shot
========================
6. request 状态机
========================
需要新增或封装 per-request SpecLink-CV 状态。
建议状态：
NORMAL
DRAFTING
VERIFY_READY
VERIFY_PREFIX
PREFIX_REJECTED
PREFIX_ACCEPTED
VERIFY_SUFFIX_READY
VERIFY_SUFFIX
COMMIT_READY
DONE_OR_NEXT_STEP
每个 request 需要保存：
SpecLinkCVState:
  request_id
  model_id
  dataset_id
  K
  draft_tokens
  draft_logprobs
  draft_margins
  draft_entropies
  a_hat
  selected_h
  verified_prefix_len
  accepted_prefix_len
  first_reject_pos
  suffix_pending
  chunk_phase
  queue_enter_time
  queue_wait_ms
  num_extra_tlm_forwards
  skipped_suffix_tokens
  fallback_reason
  correctness_debug_info
状态流程：
1. DRAFTING
   - DLM/EAGLE3 生成 K 个 draft tokens。
   - 采集 DLM confidence features。
2. VERIFY_READY
   - 根据配置选择 h。
   - 生成 VerifyChunk。
   - sync mode：进入当前 batch 的同步 chunk 逻辑。
   - async mode：进入 verification queue。
3. VERIFY_PREFIX
   - TLM exact verify 前 h 个 draft tokens。
   - 不能用 approximate classifier。
4. PREFIX_REJECTED
   - 如果 prefix 中第 j 个 token reject：
     - 使用 vLLM 原有 rejection sampler 处理该 reject。
     - accepted tokens before j 可以 commit。
     - j 之后的 suffix tokens 全部跳过。
     - request 进入 COMMIT_READY。
   - 必须保证和 one-shot EAGLE3 的结果一致。
5. PREFIX_ACCEPTED
   - 如果前 h 个全部 accepted 且 h<K：
     - 不要生成 bonus token。
     - 生成 suffix chunk。
     - 进入 VERIFY_SUFFIX_READY。
   - 如果 h=K：
     - 等价 one-shot。
     - 走原始 commit / bonus token 逻辑。
6. VERIFY_SUFFIX
   - 简化第一版：suffix 一次性 verify all。
   - 可选扩展：suffix 再次 chunk，但默认不要递归切太多，避免状态复杂。
   - suffix verification 必须 condition on accepted prefix。
   - 优先复用 prefix verification 产生的 KV；如果做不到，可以先 recompute，但必须记录 overhead，且不得宣称最终性能收益来自该版本。
7. COMMIT_READY
   - commit accepted tokens。
   - 释放或丢弃 skipped suffix。
   - 更新 vLLM request 状态，进入下一轮 decode/draft。
关键 correctness 要求：
- SpecLink-CV 是 exact prefix-gated verification。
- 所有跳过 suffix 的决策必须来自 TLM 对 prefix 的真实 reject。
- temperature=0 / greedy 下，SpecLink-CV 输出必须 match vLLM+EAGLE3 one-shot。
- 如果存在 mismatch，必须输出 request id、draft tokens、selected h、first reject pos、baseline output、SpecLink output。
========================
7. sync batch vs async queue
========================
实现两个模式，用于消融。
------------------------
7.1 sync batch mode
------------------------
当 --speclink-cv-async-queue=false：
实现一个同步批处理消融模式。
要求：
- 当前 scheduler batch 中的 SpecLink requests 尽量使用同一 chunk phase。
- 可以按照 selected_h 分 bucket，但不要跨 phase 混太多。
- 可以设置 min batch 或 max wait，但不得无限等待。
- 该模式主要用于证明 global/sync batching 会带来 head-of-line blocking 或低灵活度。
- 日志必须记录 sync wait time、batch formation delay、因等待导致的 fallback 次数。
sync mode 不要求是最终最优；它是 ablation baseline。
------------------------
7.2 async queue mode
------------------------
当 --speclink-cv-async-queue=true：
实现 verification-ready queue。
VerifyChunk 字段：
VerifyChunk:
  request_id
  chunk_id
  phase: prefix or suffix
  start_draft_pos
  chunk_len
  K
  selected_h
  a_hat_slice
  survival_prob
  reject_prob
  expected_benefit
  context_len
  age_ms
  priority
  model_id
  dataset_id
  arrival_time
  deadline_or_max_wait
每轮 scheduler：
1. 收集 VERIFY_READY 和 VERIFY_SUFFIX_READY requests。
2. 为每个 request 生成 VerifyChunk。
3. 放入 async verification queue。
4. 按 priority / bucket / roofline policy 选择一组 chunks。
5. 组成 TLM verification batch。
6. 执行 exact verification。
7. 根据结果更新每个 request 状态。
8. 对 prefix accepted 的 request 生成 suffix chunk，重新入队或直接调度。
9. 对 prefix rejected 的 request 结束当前 speculative step。
priority 建议：
Priority(c) =
  ExpectedSaving(c) / max(VerifyCost(c), eps)
  + age_weight * AgeMs(c)
ExpectedSaving(c) 对 prefix chunk 使用：
  reject_prob * suffix_cost - extra_cost
suffix chunk 的 priority：
  normal verification priority + age
必须支持防饥饿：
- 如果 AgeMs > max_queue_wait_ms，schedule now 或 fallback one-shot。
- 不允许 request 永久停在 queue 里。
========================
8. roofline-aware packing
========================
当 --speclink-cv-roofline-packing=false：
使用简单 packing：
- FIFO + priority。
- 或按 chunk_len bucket 分组。
- 不做 GPU utilization 预测。
当 --speclink-cv-roofline-packing=true：
实现轻量 roofline-aware packing。
目标：
queue 中有多个 candidate chunks 时，选择一组能带来最高 expected benefit 且预测 GPU 利用率足够高的 chunks。
需要先建立 cost model：
tools/speclink_cv/profile_verify_cost.py
对每个 model、K、chunk_len、batch size、context length 采样测量：
- TLM verifier time
- achieved tokens/s
- approximate FLOPs
- approximate KV bytes
- GPU memory
- utilization if available
- CUDA graph hit/miss if available
- launch overhead if measurable
cost model 可以先是 empirical lookup table，不必第一版就做复杂 roofline。
估计一组 chunks B 的运行时间：
T_pred(B) =
  max(
    FLOPs(B) / effective_peak_flops,
    Bytes(B) / effective_peak_bandwidth
  )
  + launch_overhead
  + metadata_overhead
或直接用 empirical lookup interpolation。
packing 规则：
1. 候选 chunks 先按 chunk_len / context_len / model bucket 分组。
2. 对每个候选组 B 估计：
   - total expected benefit
   - T_pred(B)
   - predicted utilization
   - total verify tokens
   - total seqs
3. 满足：
   - total verify tokens <= max verify token budget
   - total seqs <= max verify seq budget
   - predicted utilization >= util threshold
4. 选择：
   maximize total expected benefit / T_pred(B)
5. 如果所有小 chunk 组都不满足 utilization threshold：
   - 尝试合并相近 bucket；
   - 或等待不超过 max_queue_wait_ms；
   - 或将低收益 chunk fallback 为 one-shot；
   - 或调度 full verification。
日志必须记录：
- candidate group sizes
- selected chunks
- predicted utilization
- predicted time
- actual time
- prediction error
- fallback reason
最终报告中要说明 roofline-aware packing 的收益是否来自：
- 更高 GPU utilization；
- 更少 extra forward 浪费；
- 更低 queue wait；
- 更好 batch shape；
- 还是没有收益。
========================
9. vLLM 集成要求
========================
请在 vLLM 中实现，不要只写离线 simulator。
实施步骤建议：
1. 找到当前 vLLM EAGLE3 speculative decoding path。
2. 找到 draft/proposal 阶段，采集 DLM logits/logprobs/margin/entropy。
3. 找到 target verification 阶段，将 one-shot K-token verification 拆成 prefix chunk + optional suffix chunk。
4. 接入 request state machine。
5. 接入 sync mode 和 async queue mode。
6. 接入 roofline packing。
7. 接入 logging/profiling。
8. 确保 speclink-cv-enable=false 时，vLLM 行为与修改前一致。
尽量局部化 patch。所有修改必须保存 diff：
results/speclink_cv_TIMESTAMP/patches/vllm_speclink_cv.diff
如果完整 vLLM scheduler 改动过大，可以先实现最小可运行版本：
- wrapper-level queue；
- vLLM 内部最小 patch；
- 但必须真实调用 vLLM TLM verifier，不得只模拟。
禁止：
- 只做 trace simulator 就声称 end-to-end speedup。
- 在线用 TLM oracle 决定 chunk size。
- 改变 speculative decoding sampling 语义。
- 忽略 correctness mismatch。
- 只测 batch size 1/2/4。
========================
10. 单元测试与 smoke tests
========================
新增测试脚本：
tools/speclink_cv/test_chunk_decision.py
tools/speclink_cv/test_state_machine.py
tools/speclink_cv/test_async_queue.py
tools/speclink_cv/test_roofline_packing.py
tools/speclink_cv/test_correctness_smoke.py
必须覆盖：
1. confidence sizing off:
   - K=8 -> h=4
   - K=12 -> h=6
2. confidence sizing on:
   - all high confidence -> choose full K or larger h
   - early low confidence -> choose small h
   - benefit <= threshold -> choose full K
3. state machine:
   - prefix reject -> suffix skipped
   - prefix accepted -> suffix scheduled
   - full K -> equivalent one-shot
   - suffix verified -> commit
4. async queue:
   - chunks can arrive from different requests at different times
   - no global synchronization required
   - age timeout prevents starvation
5. roofline packing:
   - underfilled small chunks are not scheduled alone if utilization threshold not met
   - enough chunks are packed together
   - fallback path works
6. correctness smoke:
   - small MTBench subset
   - small math_reasoning subset
   - greedy decode
   - vLLM+EAGLE3 one-shot vs SpecLink-CV output exact match
测试结果输出：
02_unit_tests/unit_test_summary.md
02_unit_tests/unit_test_summary.json
========================
11. 实验数据集
========================
先测两个数据集：
1. MTBench
   - 路径和格式以 AGENTS.md / repo 脚本为准。
   - 主要用于 chat serving workload。
   - 如果没有 judge，不要声称 MT-Bench score；只报告 serving metrics 和 output consistency。
2. math_reasoning
   - 使用 AGENTS.md 指定的 math_reasoning.jsonl。
   - 如果存在 reference answer，做 answer extraction 和 exact match。
   - 如果没有可靠 reference，只做 dense / one-shot consistency，不要编造 accuracy。
每个数据集需要 split：
- calibration / tuning split
- eval split
calibration split 用于 DLM confidence calibration。
eval split 用于最终性能和 correctness。
不要在 eval split 上调 threshold、calibration、roofline 参数。
========================
12. 模型与 K
========================
模型：
- Llama3 target + corresponding EAGLE3 speculator
- Qwen3 target + corresponding EAGLE3 speculator
具体模型名、路径、config 全部从 AGENTS.md 读取。
K：
- num_speculative_tokens = 8
- num_speculative_tokens = 12
如果 K=12 对某些模型 OOM 或 vLLM/EAGLE3 不支持，记录失败原因，不要删除该配置。
========================
13. batch size / serving pressure
========================
重点测大 batch，因为 async queue 和 multi-chunk packing 需要足够多 ready chunks。
必须测：
- batch size / max_num_seqs = 8
- batch size / max_num_seqs = 16
- batch size / max_num_seqs = 32
实现方式：
- 设置 vLLM max_num_seqs / max_num_batched_tokens。
- 用 GuideLLM 或 repo benchmark 脚本产生足够并发/请求速率，使实际 batch 接近目标。
- 记录实际 average active requests / scheduled seqs，不要只记录配置值。
如果 GPU OOM：
- 保留 OOM 记录。
- 降低 max tokens 或 context length。
- 不要悄悄跳过 16/32。
========================
14. 对比方法和消融矩阵
========================
对每个 model x dataset x K x batch_size，运行以下方法。
------------------------
A. pure vLLM
------------------------
- No speculative decoding.
- No EAGLE3.
- No SpecLink.
- 作为 dense serving baseline。
method_name = pure_vllm
------------------------
B. vLLM + EAGLE3
------------------------
- EAGLE3 enabled.
- Original one-shot verification.
- SpecLink-CV disabled.
method_name = eagle3_oneshot
------------------------
C. vLLM + EAGLE3 + SpecLink-CV ablations
------------------------
SpecLink-CV enabled，跑 2 x 2 x 2 消融：
confidence sizing:
  0 = fixed half chunk
  1 = DLM confidence-guided chunk
async queue:
  0 = sync batch mode
  1 = async queue mode
roofline packing:
  0 = simple packing
  1 = roofline-aware packing
共 8 个方法：
1. cv_half_sync_simple
   confidence=0, async=0, roofline=0
2. cv_half_sync_roofline
   confidence=0, async=0, roofline=1
3. cv_half_async_simple
   confidence=0, async=1, roofline=0
4. cv_half_async_roofline
   confidence=0, async=1, roofline=1
5. cv_conf_sync_simple
   confidence=1, async=0, roofline=0
6. cv_conf_sync_roofline
   confidence=1, async=0, roofline=1
7. cv_conf_async_simple
   confidence=1, async=1, roofline=0
8. cv_conf_async_roofline
   confidence=1, async=1, roofline=1
------------------------
D. best SpecLink-CV
------------------------
从 C 中选出满足 correctness 的最佳配置。
selection criteria：
- greedy output exact match rate = 100% vs eagle3_oneshot，或解释数值非确定性并给出 token match。
- primary: output tokens/s or accepted tokens/s。
- secondary: p95/p99 ITL, p95 E2E latency。
- overhead: extra TLM forwards, queue wait, verifier time。
- robustness: model/dataset/K/batch size 平均表现。
不要只选择某一个单点最快配置；需要报告 per-scenario best 和 global robust best。
========================
15. 需要收集的指标
========================
性能指标：
- output tokens/s
- total tokens/s
- requests/s
- TTFT p50/p90/p95/p99
- ITL or TPOT p50/p90/p95/p99
- E2E latency p50/p90/p95/p99
- GPU peak memory
- actual average batch size
- actual scheduled seqs per step
- actual scheduled tokens per step
speculative decoding 指标：
- K
- acceptance rate by draft position
- weighted acceptance rate
- accepted tokens per step
- rejected tokens per step
- first reject position distribution
- prefix survival probability
- skipped suffix tokens
- skipped suffix token ratio
- extra TLM forward count
- verifier calls per generated token
- accepted tokens per TLM verifier call
chunked verification 指标：
- selected h distribution
- h by confidence bucket
- confidence enabled/disabled
- sync vs async
- queue wait p50/p95/p99
- verification queue length
- chunk age
- prefix accepted ratio
- prefix rejected ratio
- suffix scheduled ratio
- suffix skipped ratio
- fallback-to-one-shot ratio
- reason for fallback
confidence calibration 指标：
- ECE
- Brier score
- reliability diagram
- logprob vs acceptance
- margin vs acceptance
- entropy vs acceptance
- calibrated a_hat vs actual acceptance
scheduler / roofline 指标：
- predicted utilization
- actual utilization if available
- predicted TLM time
- actual TLM time
- prediction error
- packing group size
- chunks per TLM batch
- token budget utilization
- max_num_seqs utilization
- CUDA graph hit rate if available
- scheduler overhead
- launch overhead if measurable
correctness / quality：
- SpecLink-CV output exact match vs eagle3_oneshot
- token match rate vs eagle3_oneshot
- first mismatch position
- dense vs eagle3 consistency if measured
- math answer EM if reliable references exist
- MTBench judge score only if repo already supports judge; otherwise do not invent score
========================
16. 需要生成的图
========================
至少生成：
1. pure_vllm vs eagle3_oneshot vs best_speclink_cv throughput
2. same comparison p95 ITL / p99 ITL
3. same comparison p95 E2E latency
4. K=8 vs K=12 speedup
5. batch size 8/16/32 speedup
6. first reject position distribution
7. selected chunk size distribution
8. DLM confidence calibration reliability diagram
9. fixed half vs confidence-guided chunk speedup
10. sync batch vs async queue latency and throughput
11. async queue wait time distribution
12. simple packing vs roofline packing utilization
13. skipped suffix ratio vs speedup
14. extra TLM forwards vs speedup
15. ablation heatmap: confidence x async x roofline
16. best configuration by model/dataset/K/batch size
每张图都要有对应 csv。
========================
17. 运行实验的建议顺序
========================
Step 1: env check。
Step 2: pure vLLM 和 EAGLE3 one-shot smoke。
- small subset。
- Llama3 / Qwen3。
- MTBench / math_reasoning。
- K=8 first。
Step 3: correctness smoke for SpecLink-CV。
- K=8。
- batch size small。
- confidence off。
- async off。
- roofline off。
- 确认 chunked verification 输出 match eagle3_oneshot。
Step 4: confidence trace collection and calibration。
- 用 calibration split。
- 生成 DLM confidence -> acceptance calibration。
Step 5: run main ablation。
- model: Llama3, Qwen3。
- dataset: MTBench, math_reasoning。
- K: 8, 12。
- batch size: 8, 16, 32。
- methods: pure_vllm, eagle3_oneshot, 8 SpecLink-CV ablations。
Step 6: roofline profiling。
- 对不同 chunk size / batch / context length 测 verify cost。
- 评估 predicted vs actual。
Step 7: parse results and generate report。
========================
18. 最终报告必须回答的问题
========================
SPECLINK_CV_REPORT.md 必须回答：
1. pure vLLM、vLLM+EAGLE3、best SpecLink-CV 在 Llama3/Qwen3 上的端到端表现如何？
2. K=8 和 K=12 哪个更适合 SpecLink-CV？
3. batch size 8/16/32 下，SpecLink-CV 是否随着 batch 增大更有收益？
4. DLM confidence 是否能预测 TLM acceptance？
5. confidence-guided chunk size 是否优于固定 half chunk？
6. sync batch 是否造成 head-of-line blocking 或低灵活度？
7. async queue 是否改善 throughput / p95 latency？
8. roofline-aware packing 是否真的提升 GPU utilization 或减少无效小 batch？
9. SpecLink-CV 的收益主要来自：
   - skipped suffix tokens？
   - 更少 verifier time？
   - 更高 batch utilization？
   - 还是只是调度噪声？
10. extra TLM forward 和 repeated KV read 是否抵消收益？
11. correctness 是否完全保持？
12. 哪个消融组合是 global best？
13. 哪些场景不适合 chunked verification？
14. vLLM 实现中还有哪些限制，例如 KV reuse、CUDA graph、scheduler integration、chunked prefill 交互？
最终报告中必须有一张总表：
model, dataset, K, batch_size, method,
throughput, speedup_vs_eagle3,
ttft_p95, itl_p95, e2e_p95,
exact_match_vs_eagle3,
selected_h_avg,
skipped_suffix_ratio,
extra_tlm_forwards_per_request,
queue_wait_p95,
gpu_util,
fallback_ratio
========================
19. 反作弊与失败处理
========================
必须遵守：
1. 不得只写代码不跑实验。
2. 不得编造吞吐、延迟或 correctness。
3. 不得只报告成功配置。
4. OOM / crash / mismatch 必须记录。
5. 不得把模拟 speedup 写成端到端 speedup。
6. 不得在线使用 TLM accept/reject 结果决定当前 chunk size。
7. 不得在 eval split 上调 calibration 或 threshold。
8. speclink-cv-enable=false 时，行为必须与原始 vLLM+EAGLE3 一致。
9. SpecLink-CV exact mode 输出必须和 eagle3_oneshot 一致；不一致必须阻断性能 claim。
10. 如果 suffix verification 需要 recompute prefix KV，必须记录该 overhead，不得隐瞒。
11. 如果 roofline prediction 不准，必须报告 prediction error。
12. 如果 MTBench 没有 judge，不要声称 MTBench score。
13. 如果 math_reasoning 没有 reference answer，不要声称 task accuracy，只报告 consistency。
========================
20. 最终交付
========================
最终回复必须包含：
1. 修改了哪些文件。
2. 新增了哪些 flags。
3. 如何运行 pure vLLM / EAGLE3 / SpecLink-CV。
4. 单元测试是否通过。
5. smoke correctness 是否通过。
6. 主实验跑了哪些配置。
7. 哪些配置失败或 OOM。
8. 最优 SpecLink-CV 配置。
9. 相比 vLLM+EAGLE3 的吞吐提升、latency 变化、correctness。
10. DLM confidence sizing、async queue、roofline packing 三个模块各自的消融收益。
11. 所有结果目录路径。
12. 下一步还需要优化的 vLLM scheduler / KV / CUDA graph 问题。
只有在结果文件真实存在后，才能给最终总结。

这版 prompt 的核心约束是：SpecLink-CV 必须是 exact chunked verification，所有“跳过 suffix”的依据只能来自 TLM 对 prefix 的真实拒绝；DLM 置信度只负责决定“先买多少 prefix 信息”，不能直接决定接受/拒绝。