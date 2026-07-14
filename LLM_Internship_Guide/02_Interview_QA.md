# 大模型算法工程师面试题库与参考解答

> 更新基线：2026-07-14。配套：[技术学习指南](./01_Technical_Guide.md)｜[实战项目手册](./03_Practical_Projects.md)  
> 使用原则：先遮住答案口述，再录音复盘。参考答案用于建立推理框架，不用于虚构经历。

## 目录与目标

1. Python/PyTorch 与深度学习基础
2. Transformer 高频题
3. 数据、预训练与 SFT
4. LoRA、QLoRA、对齐与强化学习
5. 训练稳定性、显存与推理
6. 分布式训练
7. Prompt、评测与 RAG
8. Agent 与工具调用
9. 多模态、论文复现与开放题
10. 系统设计、项目深挖与模拟面试
11. 2024–2026 高频补充题（Q67–Q130）
12. 二十道手撕练习与项目追问九宫格

**先修知识**：至少阅读技术指南相应章节，并实际运行一个训练或应用项目。  
**学习目标**：能够对 130 个核心问题给出完整、连贯的一段式回答；遇到陌生追问时，能从公式、数据流和实验验证出发继续推理。

## 0. 使用方式

### 0.1 一段式回答原则

每道题的参考答案都整理为一个连续段落，阅读时按“定义与结论 → 核心原理 → 公式或工程细节 → 适用边界”的顺序理解。实际面试可根据时间压缩，但不要只背第一句话。

例如：“LoRA 是参数高效微调方法。它解决全参微调显存和存储成本高的问题。它冻结原权重，用两个低秩矩阵学习权重增量。代价是 rank 和注入层选择会限制容量，且训练显存仍受激活影响。”

### 0.2 项目题

按 `目标—基线—改动—评测—结果—失败—反思` 回答。所有数字必须来自自己的日志；没有测过就说“当时未测，这会是我下一步补充的指标”。

### 0.3 场景排障题

先澄清现象和口径，再按“数据 → 配置 → 数值 → 系统 → 分布式”缩小范围。每提出一个假设，都给出验证信号和下一步，而不是罗列可能性。

---

## 1. Python、PyTorch 与深度学习基础

### Q1：PyTorch 自动微分是怎么工作的？

PyTorch 在前向时动态记录由 Tensor 和算子构成的计算图；`backward()` 从标量 loss 反向应用链式法则，把梯度累积到叶子参数的 `.grad`。默认图在反向后释放，因此每次更新前要清梯度，重复反向需保留或重建图。`requires_grad` 决定是否跟踪；`grad_fn` 指向反向节点；非标量输出调用 backward 时需传入向量—Jacobian product 的上游梯度。in-place 操作可能破坏反向需要的版本。训练循环是 zero grad → forward → loss → backward → optimizer step。

### Q2：`model.train()`、`model.eval()` 和 `torch.no_grad()` 有什么区别？

`train/eval` 切换 Dropout、BatchNorm 等模块行为，不控制计算图；`no_grad`/`inference_mode` 才关闭梯度跟踪。验证时通常二者同时使用。`eval()` 不冻结参数；若忘记 eval，Dropout 使结果随机、BN 继续用 batch 统计。`inference_mode` 比 `no_grad` 更激进地关闭版本跟踪，适合纯推理。

### Q3：Adam 和 AdamW 为什么不同？

AdamW 将 weight decay 从梯度中的 L2 项解耦，直接按权重衰减，再做 Adam 自适应更新；这样衰减不被每参数自适应学习率扭曲。Adam 保存一阶、二阶矩，显存成本高。通常对矩阵权重衰减，不对 bias 和 norm 参数衰减。学习率、beta、epsilon、warmup 都会影响稳定性。

### Q4：如何保证实验可复现？

固定代码、数据、配置、依赖、随机种子和评测流程；记录硬件与框架版本；必要时启用确定性算法。但分布式规约、某些 CUDA kernel 和数据顺序仍可能带来差异，所以还应多种子报告均值和方差。保存 commit、数据 hash、模型 revision、chat template、生成参数和日志；严格 resume 还要恢复 optimizer/scheduler/scaler/RNG/sampler。

### Q5：训练 loss 下降，验证效果却变差，怎么处理？

先确认训练/验证处理一致且无数据泄漏，再看是否过拟合、目标与指标错位或评测解码变化。可减少 epoch、降低容量/学习率、加正则和高质量数据，并用错误分桶判断能力回退发生在哪里。LLM 微调还要检查 prompt loss mask、chat template、生成参数、灾难性遗忘和答案长度偏置。训练 NLL 与任务正确率并非单调对应。

---

## 2. Transformer 高频题

### Q6：写出 Self-Attention 公式并解释为什么缩放

`softmax(QK^T/√d_k + M)V`。若 Q、K 各维近似独立同方差，点积方差随 `d_k` 增大；除以 `√d_k` 把尺度归一，避免 softmax 饱和与梯度过小。说明形状 `[B,H,T,Dh]`，分数 `[B,H,T,T]`；mask 屏蔽 padding 或未来位置。softmax 常在 FP32 做以稳定数值。

### Q7：Multi-Head Attention 为什么有效？头越多越好吗？

多头把表示投影到多个子空间，让模型同时学习不同位置与关系模式；但固定 `d_model` 时头越多，单头维度越小，且通信/内核效率可能变差，所以不是越多越好。总复杂度量级仍约 `O(T²d)`；部分头可能冗余。分组查询注意力GroupQueryAttention（GQA）/多查询注意力MultiQueryAttention（MQA） 主要减少 KV heads 的数量，从而降低推理 Cache 和带宽。

### Q8：Attention 的时间和空间复杂度是什么？

QK 和 AV 的主计算是 `O(BT²d)`，显式注意力矩阵空间是 `O(BHT²)`；投影还有 `O(BTd²)`。短序列大隐藏维时投影可能显著，长序列时二次项主导。训练存激活；自回归有 KV Cache 后，每个新 token 对历史做线性注意力，但整个生成过程对序列长度累计仍有二次计算趋势。FlashAttention 降低 HBM 读写和中间矩阵存储，不改变标准精确注意力的数学结果和理论 FLOPs 阶。

### Q9：Pre-Norm 和 Post-Norm 有什么区别？

Post-Norm 在残差相加后归一化；Pre-Norm 先归一化再进子层并加回残差。Pre-Norm 给梯度提供更直接的残差路径，通常更容易训练深网络，但输出尺度和最终 norm 设计也需配套。写出 `x+F(norm(x))` 与 `norm(x+F(x))`；不能只从单层推断所有架构效果。现代 LLM 还常用 RMSNorm。

### Q10：RoPE 的直觉是什么？如何扩展上下文？

RoPE 按不同频率旋转 Q/K 的二维分量，使它们的点积自然依赖相对位置差。扩上下文可做位置插值或频率缩放，但只改配置可能产生分布外位置，需长文本继续训练和专项评测。旋转保持向量范数；低频承载长程、高频承载局部信息。扩展要同时考虑注意力显存、训练长度分布和“lost in the middle”。

### Q11：KV Cache 是什么，显存如何估算？

自回归生成时缓存历史 token 每层的 K/V，避免每步重复计算。粗略元素数是 `2 × layers × batch × sequence × kv_heads × head_dim`，再乘 dtype bytes；GQA/MQA 通过减少 `kv_heads` 降低 Cache。并发服务还有 block/page 管理、碎片和 prefix cache。权重量化不必然压缩 KV Cache，Cache dtype 是独立配置。

### Q12：BERT、LLaMA/Qwen、T5 的结构和目标差异？

BERT 是双向 Encoder，用 MLM 学表示；LLaMA/Qwen 类是 causal Decoder，用 next-token prediction 做生成；T5 是 Encoder-Decoder，把任务统一成 text-to-text。结构选择决定 attention mask、预训练目标和适用任务。Encoder 可同时看两侧上下文，适合判别/抽取；Decoder 统一生成接口，易扩展对话和 Agent；Encoder-Decoder 对条件输入编码更明确。

### Q13：为什么现代 LLM 常用 SwiGLU？

SwiGLU 用一条门控分支 `silu(gate(x))` 调制另一条 `up(x)`，再投影回隐藏维，通常比简单 ReLU/GELU FFN 有更强表达。为控制参数量，中间维度会相应调整。它引入三个线性映射而非普通 FFN 两个；参数和 FLOPs 比较必须在等预算下进行。

---

## 3. 数据、预训练与 SFT

### Q14：Tokenizer 为什么是模型的一部分？

Tokenizer 决定文本到 token id 的映射、序列长度、special tokens 和 chat template；Embedding 行与词表一一对应。随意替换会让 id 语义错位，即使词表大小一样也不能直接兼容。新增 token 后需 resize embedding，并训练新行；比较不同模型 token 效率时要按语言和领域统计。

### Q15：Causal LM 的标签如何构造？

模型在位置 `t` 的输出预测下一个 token；框架通常在 loss 内部 shift logits/labels。padding、用户提示或不参与监督的位置用 `-100`，CrossEntropy 会忽略。必须读具体模型/Trainer 实现，避免手动 shift 两次。Chat SFT 常只对 assistant 内容算 loss。

### Q16：如何构建高质量 SFT 数据？

先定义能力和 rubric，再采集/生成、去重、过滤、事实校验、格式校验和安全检查；按来源或实体切分防泄漏；统计主题、长度、难度和拒答分布。少量高质量且覆盖明确的数据通常胜过大量噪声模板。保留 provenance、license、版本；抽样人工审核；用规则+模型初筛但防 judge 偏差；对多轮数据检查角色顺序和上下文依赖。

### Q17：Packing 为什么能加速？有什么坑？

把多个短样本拼到固定长度，减少 padding，使每次矩阵计算处理更多有效 token。坑是样本边界、position ids、causal mask 和 loss mask 若处理错误，会产生跨样本信息泄漏。区分简单 concatenate 与 sequence packing；记录 effective tokens/s 而非 samples/s。

### Q18：预训练、继续预训练和 SFT 有什么区别？

预训练在大规模通用语料上学习语言/世界模式；继续预训练用领域无标注文本改变知识和分布；SFT 用指令—答案示范改变行为和任务格式。三者常用相同 next-token loss，但数据分布、学习率、目标和风险不同。领域知识更新不一定靠 SFT 最有效；继续预训练可能损害通用能力，需要混入通用数据和回归评测。

---

## 4. LoRA、QLoRA、对齐与强化学习

### Q19：LoRA 原理、参数量和初始化？

冻结 `W`，学习 `ΔW=(α/r)BA`。若 `W` 是 `d_out×d_in`，新增参数 `r(d_in+d_out)`。常把一矩阵随机初始化、另一矩阵初始化为零，使训练开始时 `ΔW=0`，保持基座初始输出。rank 是容量不是直接压缩率；注入 q/v 或 all-linear 影响参数和效果；推理可动态加载 adapter，或 merge 到浮点基座减少额外计算。

### Q20：QLoRA 为什么省显存？质量风险在哪里？

冻结基座以 4-bit 存储，仅训练 LoRA；NF4 适合近似正态权重，double quant 进一步压缩量化常数，计算一般用 BF16。风险是量化误差、硬件/kernel 兼容和某些层敏感，且激活显存不会按 4 倍下降。优化器只维护 adapter 状态；加载后通常调用 k-bit training preparation；训练 extra parameters，不是直接更新 4-bit 权重。

### Q21：SFT 和 DPO 各解决什么问题？

SFT 模仿单个理想回答，建立基本指令行为；DPO 用 chosen/rejected 对学习相对偏好，并通过 reference policy 约束偏离。DPO 通常在已有 SFT policy 上做，不能替代所有能力学习。DPO 的 beta 控制偏好强度；数据质量、长度偏置和 reference 选择很关键。DPO 训练简单于显式 RM+PPO，但也只优化给定偏好分布。

### Q22：RLHF 的完整流程和难点？

通常先 SFT，再采集偏好训练 Reward Model，最后用 PPO 等 RL 方法优化 policy，并加 KL 约束参考模型。难点是四类模型/服务的资源编排、reward hacking、rollout 成本、奖励尺度和训练稳定性。PPO 用 advantage 和 clipped objective 限制更新；价值模型估计回报。要评估 helpfulness、safety、能力回退和 KL。

### Q23：DPO 的 beta 增大意味着什么？

在常见 DPO 参数化中，beta 影响相对 reference 的偏好优化尺度，可理解为偏离约束/奖励温度的一部分；具体“越大越保守还是越激进”要结合实现公式说明，不能脱离 loss 定义背结论。直接写当前实现的 logit：`beta * [(logπw-logπref,w)-(logπl-logπref,l)]`；beta 变大使 sigmoid logit 绝对值放大和梯度分布变化。最终行为还受 lr、数据和 label noise 影响。

### Q24：PPO 为什么要 clip？

重要性比率 `r=π_new/π_old` 若变化过大，会让基于旧轨迹的更新不可靠。PPO 对 surrogate objective 的 ratio 做裁剪，限制单次策略更新幅度，提高稳定性。clip 不等于严格 trust region；还常监控/惩罚 KL、value loss、entropy。优势估计质量和 reward normalization 同样关键。

### Q25：Agentic RL 与普通偏好优化有什么不同？

普通偏好优化多在固定 prompt-response 数据上；Agentic RL 让 policy 在环境中多步调用工具并根据任务结果获得奖励。它要处理状态、长时 credit assignment、环境错误、稀疏奖励和高成本 rollout。可用规则验证器、单元测试、环境成功信号；需要轨迹级 replay/过滤和防 reward hacking。离线数据与 on-policy 轨迹分布不同。

### Q26：什么是 On-Policy Distillation？

核心是在学生当前策略访问到的状态/序列分布上，让教师提供软目标或反馈，从而减少纯离线教师数据与学生实际分布的偏移。不同论文把 OPD 用在 reasoning、policy 或 sequence distillation，面试时应先确认具体论文。与纯 imitation 的区别是数据来自当前 policy；与 RL 的区别是学习信号主要来自教师分布/目标而非环境标量奖励。代价是在线采样和教师推理昂贵。

---

## 5. 训练稳定性、显存与推理

### Q27：训练突然出现 NaN，如何排查？

先定位首次 NaN 的 step 和 tensor，再检查该 batch 数据、loss mask/除零、学习率和 warmup、FP16 overflow、grad norm、自定义算子。用 anomaly detection 或 hooks 缩小层，并用 BF16/FP32 小跑验证数值假设。分布式要找最先异常 rank；检查 scaler 是否连续跳 step、logits 是否 inf、空 label 导致 loss 非法。修复后用同一 batch 回归。

### Q28：OOM 时怎么优化？

先区分加载 OOM、前向 OOM、反向 OOM 和保存 OOM，并测峰值。优先减 micro-batch/长度、gradient checkpointing、混合精度；再用 LoRA/QLoRA、FlashAttention、ZeRO/FSDP 或 offload。保持有效 batch 时用梯度累积。加载 OOM 多为权重/设备映射；反向多为激活/梯度；优化器 step 多为 states；推理长上下文多为 KV Cache。碎片问题需看 reserved/allocated，不能迷信清 cache。

### Q29：梯度累积是否等价于大 batch？

在样本、loss 归一、无跨 batch 状态且优化器只在累积后 step 时，梯度近似等价；但 Dropout 随机性、BatchNorm、序列长度归一、梯度裁剪位置和分布式通信会造成差异。loss 通常除 accum steps；scheduler 以 optimizer step 还是 micro-step 计数要一致；DDP 可在非更新步使用 `no_sync` 减少通信。

### Q30：FP16 与 BF16 怎么选？

FP16 尾数更多但指数范围小，常需 GradScaler；BF16 指数范围接近 FP32，训练更稳但有效精度较低。硬件支持 BF16 时 LLM 训练常优先 BF16，仍要监控敏感算子和 loss。参数 dtype、计算 dtype、累积 dtype 可不同；softmax、norm、reduction 常提升精度。选择还看 GPU 架构和 kernel。

### Q31：量化为何能加速/为何有时不加速？

量化减少权重内存和带宽，若有高效低比特 kernel 可提升吞吐；但若运行时频繁反量化、算子不支持、batch 太小或瓶颈不在权重带宽，可能只省显存不加速。区分 weight-only、W8A8、KV quant；考虑校准、outlier 和精度回退。

### Q32：如何评测推理服务？

同时测 TTFT、TPOT、端到端 P50/P95/P99、requests/s、tokens/s、并发、失败率、显存和质量；固定输入/输出长度分布，并区分冷启动与稳态。离线吞吐不能代表在线 SLO；continuous batching 提吞吐可能增加单请求等待。要画并发—吞吐—尾延迟曲线。

---

## 6. 分布式训练

### Q33：DDP 的基本流程是什么？

每个进程持有完整模型，`DistributedSampler` 分不同数据，反向时按 bucket 对梯度做 All-Reduce，使各 rank 更新一致。通常一 GPU 一进程，用 `torchrun` 启动。DDP 注册 autograd hooks，在梯度 ready 后通信，可与剩余反向重叠。总 batch 是 per-rank batch × world size × accumulation。

### Q34：ZeRO 1/2/3 分别分片什么？

Stage 1 分 optimizer states，Stage 2 再分 gradients，Stage 3 再分 parameters；stage 越高每卡冗余越少，但前后向参数收集等通信更多。ZeRO 与 DP 语义兼容；offload 可放 CPU/NVMe。选择依据是模型是否放得下、网络、吞吐目标和 checkpoint 复杂度。

### Q35：TP 和 PP 有什么区别？

TP 在一层内部切矩阵，需要频繁低延迟 collective；PP 按连续层切 stage，用 micro-batch 流水执行，通信激活并有 bubble。TP 适合单层放不下，PP 适合按层分割大模型。常把 TP 放在节点内高速互联，PP 跨节点；再叠 DP。PP bubble 与 stage 数和 micro-batch 数相关，还要做层负载均衡。

### Q36：什么是 EP 和 CP？

EP 将 MoE experts 分布到不同设备，token 按路由发生 All-to-All；CP 把长序列上下文维度分片，降低每卡长上下文激活/注意力负担。二者分别针对 expert 容量和序列长度。EP 要关注负载均衡和掉 token；CP 要处理注意力所需的跨设备 K/V 通信。

### Q37：多卡吞吐没有线性增长，怎么定位？

先算 scaling efficiency，再分解 data wait、compute 和 collective；检查 batch 是否过小、网络拓扑、bucket、同步点、负载不均和 CPU/I/O。用 profiler 看通信是否与反向重叠。对比单卡；逐级扩容；检查 NCCL 日志；固定 global batch 与扩大 global batch 是不同实验口径。

### Q38：分布式训练卡死有哪些常见原因？

某 rank 异常退出、不同 rank 进入不同 collective/次数、数据长度不一致、条件分支导致 backward 图不同、网络/NCCL 配置问题。先找最早异常 rank，而不是只看卡住的 rank。开启分布式 debug/NCCL 日志，设置超时，打印 rank/step，最小化数据；确认所有进程保存/验证时的 barrier 逻辑。

---

## 7. Prompt、评测与 RAG

### Q39：Prompt Engineering 和 Context Engineering 的区别？

Prompt Engineering 主要设计一次模型调用的指令和示例；Context Engineering 管理进入上下文的所有信息，包括系统规则、历史、检索证据、工具定义、记忆、压缩和 token 预算。后者是系统级数据流。上下文要按相关性、可信度、时效和优先级选择，处理冲突并防注入。

### Q40：temperature 和 top-p 怎么影响生成？

temperature 缩放 logits，改变分布尖锐程度；top-p 只保留累计概率达到 p 的最小候选集合再采样。两者同时变化会耦合，评测时应固定。贪心适合可判定任务；开放生成需在多样性和稳定性间权衡。temperature 0 的具体实现仍依服务框架。

### Q41：如何建立可信的 LLM 评测？

先定义能力和 rubric，构建无泄漏、分层测试集；固定 prompt/template/解码；组合规则指标、人工盲评和校准后的 LLM judge；报告样本数、置信区间及错误分桶。保留逐样本输出，避免只看总分；对 judge 做位置交换、顺序随机和人工一致性检查。

### Q42：RAG 的完整链路是什么？

离线做解析、清洗、切分、embedding 和索引；在线对 query 改写，做 dense/BM25 混合召回、过滤和 rerank，再组装 context 生成带引用答案；最后分层评测检索、生成和端到端。元数据权限过滤应尽量在检索侧保证；索引需版本和增量更新。

### Q43：Chunk size 如何选择？

太小缺上下文、太大主题混杂且挤占 token；应根据文档结构和问题证据跨度，用 token 统计设计候选值，并在固定测试集上比较 Recall、rerank、faithfulness、延迟与成本。标题继承、父子块和语义切分能兼顾粒度；overlap 解决边界但增加重复。

### Q44：Dense、BM25 和 Reranker 怎么配合？

Dense 擅长语义相似，BM25 擅长专名、编号和精确词；先并行召回并融合，再用 cross-encoder/LLM reranker 精排少量候选，通常比单路稳，但增加延迟。融合可用 RRF；reranker 只排序候选，召回阶段没找到的文档救不回来。

### Q45：RAG 答错如何定位？

先看 gold evidence 是否被解析和索引；再看是否召回、是否被 rerank/过滤丢掉、是否进入最终 context；若证据存在再检查冲突、prompt 和生成。按层定位而不是先换模型。把错误分成 ingestion、retrieval、ranking、packing、generation、citation、freshness。每层有独立日志和指标。

### Q46：Faithfulness 和 Answer Correctness 有何区别？

Faithfulness 看回答中的陈述能否由给定 context 支持；Correctness 看回答相对真实答案是否正确。回答可忠实于错误/过时 context 却不正确，也可凭参数记忆答对但不忠实于证据。因此还要测 retrieval、citation 和 freshness。LLM judge 的这两个 rubric 要分开。

---

## 8. Agent 与工具调用

### Q47：什么才算 Agent？

Agent 不只是一次 LLM 调用，而是模型基于状态选择动作、调用工具、观察结果并循环直到终止的系统；还需要权限、错误恢复、记忆和可观测性。固定可预测流程更适合普通 workflow。模型是决策组件，框架负责状态和控制。Agent 的自主程度应与风险匹配。

### Q48：ReAct 的优缺点？

ReAct 交替进行推理、动作和环境观察，使模型能根据工具结果调整计划；优点是可处理开放多步任务，缺点是成本高、会循环、错误观察会传播，并存在 Prompt Injection 风险。生产上设置最大步数、工具 schema、超时、重试和终止条件；记录简化决策依据与工具轨迹。

### Q49：如何防止 Agent 无限循环？

设置最大步数/token/时间/费用，检测重复状态或相同工具参数，给工具结果明确成功码，使用任务完成验证器，并允许人工中止。达到预算时返回可解释的部分结果。状态机中显式定义 terminal nodes；失败重试用指数退避且有上限；幂等写操作避免重复副作用。

### Q50：Agent 的短期和长期记忆如何设计？

短期记忆维护当前任务消息和状态，通过裁剪/摘要控制 token；长期记忆存跨会话事实/偏好，需写入门控、来源、权限、更新和遗忘。检索长期记忆时还要防用户串线和污染。区分 semantic/episodic/procedural；摘要可能丢约束，关键字段应结构化保存。

### Q51：工具调用失败如何处理？

先在执行前做 schema、权限和业务校验；执行时设超时、幂等键和有限重试；把结构化错误返回给控制器决定改参、换工具或终止；高风险写操作需人工确认。区分 transient/permanent/user error；不要把堆栈和密钥原样送回模型。

### Q52：LangChain、LangGraph、LlamaIndex、AutoGen 如何选？

LangChain 提供通用模型/检索/工具组件；LangGraph 适合显式状态图、持久化和人工介入；LlamaIndex 强在数据摄取与索引；AutoGen 强在对话式单/多 Agent。先按数据流和可靠性需求选，不为框架而框架。评估接口稳定性、可测试性、可观测性、团队熟悉度和锁定成本；核心业务接口应与框架解耦。

### Q53：如何评测 Agent？

端到端测任务成功率，同时分解 tool selection、参数正确率、步骤数、重复调用、成本、延迟和安全违规；保存轨迹并按 planning/tool/observation/memory/termination 归因。工具用 mock/sandbox 保证确定性；成功判定优先规则/环境验证器，LLM judge 只补充开放质量。

---

## 9. 多模态

### Q54：视觉 token 如何接入 LLM？

视觉编码器把图像切成 patch 并输出视觉特征，projector/adapter 将其映射到 LLM embedding 维度，再按模板与文本 token 拼接或交互，LLM 生成答案。不同模型会压缩视觉 token、支持动态分辨率或 cross-attention；视觉 token 数直接影响序列长度和显存。

### Q55：为什么先冻结视觉编码器和 LLM，只训练 projector？

两端已有稳定表示，只训练 projector 成本低、能先建立模态空间映射，也减少小数据破坏基座能力。领域视觉差异大时再逐步解冻，并给不同模块设不同学习率。三阶段常见但非所有模型固定配方；解冻视觉塔增加激活/优化器显存。

### Q56：多模态数据有哪些常见问题？

坏图/重复、图文不匹配、答案在图中不可见、OCR 泄漏、分辨率不足、模板错误、多图顺序错和类别不平衡。要做可视抽检和遮图/换图反事实测试。动态分辨率造成每样本 token 数不同，影响 batch 和负载均衡。

### Q57：如何评测视觉语言模型幻觉？

构建对象存在性、属性、计数、空间关系、OCR 等分桶集；要求答案有可验证定位/证据；做换图、遮挡和不存在对象的负样本，统计错误陈述和拒答。自动 benchmark 配合人工审核；LLM judge 可能也看错图，需校准。

### Q58：InternVL、Qwen-VL、LLaVA 有何差异？

它们都可抽象为视觉编码—对齐—语言生成，但视觉 backbone、projector、动态分辨率、视觉 token 压缩、数据配方和训练阶段不同。准确比较必须指定版本并查看论文、config 和模型卡。面试可先给共同抽象，再比较目标岗位使用版本；不要用旧版结论覆盖整个家族。

---

## 10. 论文复现与开放题

### Q59：你如何复现一篇论文？

先提炼问题、核心公式和一个最小可证伪结果；固定官方代码 commit，跑作者 checkpoint；对齐数据、预处理、模型、训练和评测；复现主结果后再做一个消融，并如实报告差异。建立 paper card 和 reproduction matrix；用中间统计定位偏差；记录硬件与随机种子。

### Q60：你怎么看世界模型？

世界模型学习环境状态和动态，用于预测未来、规划或 imagined rollouts。关键不只是生成逼真观测，而是 latent state 是否支持可控、因果一致和长时规划；误差会随 rollout 累积。说明 state/dynamics/observation/policy；语言模型的 next-token prediction 可学习部分世界规律，但不自动等同于具备可行动的世界模型。

---

## 11. 系统设计题

### Q61：设计一个企业内部知识助手

我会离线完成权限感知的文档解析、切分和版本化索引，在线请求经过鉴权、query rewrite、BM25 与 dense 混合检索、rerank、context packing 和带引用生成，并为全链路记录 trace、分层评测和反馈回流。设计前先澄清用户量、文档量、更新频率、SLO、权限和答案风险；数据面包含对象存储、解析队列、chunk/version、双路索引和删除传播，请求面包含 gateway、tenant/auth、retriever、reranker、LLM server 与 citation validator，可靠性方面加入缓存、超时、降级、幂等、审计和 Prompt Injection 防护。敏感文档权限必须在查询时由检索层实时或短 TTL 校验，最终同时监控 Recall@K、faithfulness、citation、任务成功率、P95 延迟和成本。

### Q62：设计一个可执行操作的 Agent

用显式状态图把读取、计划、校验、执行、确认和终止分开；工具最小权限、结构化 schema、幂等和审计；写操作前做 policy check，高风险动作人工确认；用任务级验证器判断真实成功。持久化 state/checkpoint，错误分类重试，预算限制，sandbox，trace 脱敏。

### Q63：如何选模型和部署方式？

先从任务质量、安全、语言/模态、上下文和合规设门槛，再比较开源自部署与 API 的延迟、吞吐、成本和运维。用真实流量分布压测，而不是只看参数量/榜单。小模型路由、量化、batching、cache、fallback；模型升级需要 shadow/canary 与回归集。

---

## 12. 项目深挖与模拟面试

### 12.1 90 秒自我介绍模板

```text
我主要准备大模型训练与应用落地。基础方面，我用 PyTorch 从零实现过
Transformer，并做过 [实际序列长度/卡数] 下的显存和吞吐实验。
后训练方面，我在 [真实数据集] 上完成了 QLoRA/SFT，并用 [真实指标]
比较基座和微调模型。应用方面，我做了带分层评测的 RAG 和有状态 Agent，
重点不是只跑通 Demo，而是记录召回、引用、工具调用与失败恢复。
我希望在实习中继续做 [与岗位匹配方向]。以上项目代码、配置和日志都可复现。
```

删除未真实完成的句子。不要用“精通”替代证据。

### 12.2 项目追问树

面试官通常沿以下路径深挖：

1. **为什么做**：任务、用户、约束是什么？
2. **为什么这样选**：为何这个基座、数据、框架、指标？替代方案呢？
3. **你具体做了什么**：指出代码模块和关键参数，而非“我们”。
4. **结果可信么**：基线、对照、样本数、方差、数据泄漏？
5. **失败过什么**：现象、假设、证据、修复、回归测试？
6. **扩大十倍怎么办**：数据、GPU、索引、并发、成本和团队协作？

### Q64：请介绍你的 QLoRA 项目

“目标是让 `[模型]` 在 `[领域]` 上改善 `[能力]`。我先用固定 `[N]` 条测试集测 base，随后构建 `[数据量]` 条 SFT 数据并做泄漏/长度检查。训练采用 NF4 4-bit 基座、BF16 compute、LoRA rank `[r]`，注入 `[layers]`；选这些参数是基于显存和容量对照。最终 `[指标]` 从 `[A]` 到 `[B]`，但 `[错误类型]` 仍明显。一次失败是 `[真实失败]`，我通过 `[证据]` 定位到 `[原因]` 并修复。部署用 `[服务]`，实测 `[并发口径]` 下 P95 `[数值]`。”

### Q65：请介绍你的 RAG 项目

“数据是 `[文档类型/规模]`，难点是 `[专名/更新/权限]`。我建立 `[N]` 条带 gold evidence 测试集，先测 BM25/dense 基线，再加入 RRF 与 reranker。检索 Recall@K 从 `[A]` 到 `[B]`，端到端 faithfulness `[C]`，P95 `[D]`。最主要错误来自 `[层]`；我通过 `[改动]` 修复。系统保存 chunk id 和引用，能从错误答案追到原文。”

### Q66：请介绍你的 Agent 项目

“任务需要 `[工具集合]` 的多步协作。我用 LangGraph 把 `[节点]` 显式化，state 中保存 `[字段]`，每个工具有 schema、timeout 和结构化错误。测试集 `[N]` 条，成功率 `[A]`，工具参数正确率 `[B]`，平均步数 `[C]`。我加入重复调用检测与 max steps 后，循环失败从 `[D]` 降到 `[E]`。写操作使用人工确认和幂等键。”

### 12.3 行为题中的技术表达

**“讲一次失败”**：选真实技术失败，讲清信号和决策，不要把“太追求完美”包装成失败。  
**“与同学意见不同”**：用共同指标做小实验，而非争框架偏好。  
**“时间不足”**：先交付可测基线和最大风险验证，再迭代；说明砍掉了什么。  
**“不会的技术”**：先澄清边界，说已知相邻概念和验证路径，不现场编造。

### 12.4 一小时模拟面试脚本

| 时间 | 内容 |
|---|---|
| 0–5 分钟 | 自我介绍与岗位动机 |
| 5–15 分钟 | Transformer 白板题（Q6–Q13） |
| 15–25 分钟 | 微调/对齐（Q19–Q26） |
| 25–35 分钟 | 项目深挖（Q64–Q66 选一） |
| 35–45 分钟 | RAG/Agent 场景排障 |
| 45–53 分钟 | 分布式或多模态专项 |
| 53–58 分钟 | 系统设计权衡 |
| 58–60 分钟 | 反问面试官 |

反问示例：团队当前更关注预训练、后训练还是应用评测？实习生交付如何评估？训练和线上错误分析各自有哪些现成基础设施？

## 13. 2024–2026 高频补充题（Q67–Q130）

### 13.1 高频趋势与使用说明

下表来自 `interview/` 中近年面经的定性归纳。不同仓库存在转载、同义题和样本偏差，因此“上升/高频”只用于安排复习优先级，不代表公司或岗位的统计概率；答案以论文、规范和官方文档为准。

| 年份 | 更常出现的主题 | 面试准备重点 |
|---|---|---|
| 2024 | Transformer、LoRA/SFT、RAG、推理基础、项目复现 | 公式能手推，项目能跑通并有基础评测 |
| 2025 | 数据与评测、项目深挖、DPO/GRPO、MoE、部署与分布式、Agent | 不只讲组件，能说明数据、基线、消融、性能瓶颈 |
| 2026 | Agent 工程与评测、RL 对齐、RAG 可靠性、推理优化、VLM | 状态/安全/稳定性与线上指标成为重点；前沿名词必须限定版本 |

高频不等于只背答案。项目类问题中的占位符只能替换为自己真实完成的内容和实测数字。

### 13.2 数学与机器学习（8 题）

### Q67：推导 Softmax 的梯度；为什么与交叉熵合用后是 `p-y`？

`∂p_i/∂z_j=p_i(δ_ij-p_j)`；对 one-hot 交叉熵用链式法则后，logits 梯度为 `p-y`。实现用 fused cross-entropy/log-softmax，避免先 softmax 再 log 的溢出。分别推 `i=j` 与 `i≠j`，再说明梯度和为 0，给所有 logits 同加常数不改变概率。标签平滑后 `y` 换为平滑分布。写 Jacobian `J=diag(p)-pp^T`；用 autograd 与手算结果做 `assert_close`。

### Q68：交叉熵、NLL 和 KL 散度是什么关系？

`H(q,p)=-E_q log p=H(q)+KL(q||p)`；真实分布固定时，最小化交叉熵等价于最小化前向 KL。one-hot 分类的 CE 就是正确类别 NLL。NLL 是最大似然的负形式；KL 非对称且不是距离。蒸馏中 teacher 是软 `q`，温度改变暗知识和梯度；语言模型按有效 token 平均 NLL。`KL(q||p)=Σq log(q/p)`；perplexity 是平均 token NLL 的指数。

### Q69：L1、L2 正则与 weight decay 有什么区别？

L1 倾向稀疏，L2 对大权重惩罚更强；SGD 下 L2 与 weight decay 可等价，但 Adam 的自适应缩放会破坏等价，AdamW 将衰减与梯度更新解耦。写出 L1 次梯度和 L2 梯度；解释为何常不衰减 bias/norm。正则是控制泛化的一种手段，不能替代数据划分与早停。`w←w-η AdamGrad-ηλw` 是 AdamW 直觉形式。

### Q70：AdamW 每一步做什么？何时不如 SGD？

AdamW 对梯度维护 EMA 一阶矩 `m` 和二阶矩 `v`，做偏差修正，用 `m/(sqrt(v)+eps)` 自适应更新，再解耦衰减参数。它收敛快、对尺度不敏感，但状态显存大，泛化与吞吐不必优于 SGD。解释训练初期 EMA 从零开始为何需 bias correction；`eps` 是数值稳定项；LLM 常用 AdamW 是经验与生态选择。写 `m_t=β1m_{t-1}+(1-β1)g_t`、`v_t=β2v_{t-1}+(1-β2)g_t²`。

### Q71：为什么需要 learning-rate warmup？步数如何选？

训练初期参数和优化器统计未稳定，大 LR 容易造成不可逆更新；warmup 逐步升 LR，之后接 decay。按 optimizer update steps 计算，而非 micro-batch 次数。结合 batch、预训练/微调、模型规模选择 warmup ratio/steps；小数据若 warmup 太长，模型几乎没有在目标 LR 学习。先观察 loss spike、grad norm，而非套固定百分比。`updates=ceil(num_batches/grad_accum)×epochs`。

### Q72：如何区分梯度爆炸、消失和低精度溢出？

记录首次异常 step 的 global/per-layer grad norm、激活与 loss scale。爆炸表现为 norm 激增，消失为早层长期接近零，FP16 溢出常伴 GradScaler 降 scale/非有限值；三者可能共存。先查数据、label/mask、LR 和自定义算子，再查精度；clip 是保护措施，不是根因修复。用 anomaly detection 只做小规模定位，因开销大。`clip_grad_norm_` 必须在 `scaler.unscale_` 后。

### Q73：AUC、Precision、Recall、F1 如何选？

Precision 关注预测为正的纯度，Recall 关注正例覆盖，F1 是二者调和平均；ROC-AUC 衡量全阈值排序，极不平衡时 PR-AUC 更能反映正类质量。选型取决于 FP/FN 成本和是否需要固定阈值。说明 macro/micro/weighted；阈值应在 validation 上选，test 只做最终报告；概率还需校准。`F1=2PR/(P+R)`；报告 confusion matrix 和置信区间。

### Q74：训练集效果好、验证集差一定是过拟合吗？

不一定；也可能是划分分布不同、预处理不一致、验证标签噪声、数据泄漏反向或 eval 模式错误。先验证管线，再谈过拟合。做 train/valid 长度、类别、来源、时间分布对比；小样本人工审计；确认 `eval()`、模板和解码一致。确认为过拟合后再早停、正则、数据增强、降容量/epoch。学习曲线看数据量增大时 train/valid gap 如何变化。

### 13.3 Attention、MoE 与模型架构（10 题）

### Q75：MHA、MQA、GQA 如何比较，KV Cache 能省多少？

MHA 每个 query head 有独立 K/V；MQA 所有 query heads 共用一组 K/V；GQA 每组 query heads 共用一组 K/V。Cache 与 `num_kv_heads` 线性相关，因此从 `Hq` 降到 `Hkv`，K/V 部分约降为 `Hkv/Hq`。写 `[B,Hq,T,Dh]` 与 `[B,Hkv,T,Dh]`；decode 常受 KV/权重带宽影响，因此共享 KV 有利。质量、kernel 支持和 checkpoint 转换需实测。`bytes=2BLTHkvDh×dtype_bytes`。

### Q76：MLA 与 GQA 是一回事吗？

不是。GQA 在 head 维度共享较少 K/V heads；MLA 把 K/V 表示压到潜在低维缓存，再在计算时恢复/吸收投影，并专门处理带 RoPE 的部分。二者都省 KV Cache，但参数化不同。说明 MLA 的目标是低秩联合压缩，不应只按 `num_kv_heads` 描述；具体矩阵和 decoupled RoPE 必须引用对应论文/实现。比较缓存口径时用实际 config 与每层缓存张量，不凭模型总参数猜。

### Q77：FlashAttention 为什么快？它是近似算法吗？

它通常计算与标准 attention 等价的结果，通过 SRAM tiling 和 online softmax 避免把完整 `T×T` score/probability 写回 HBM，减少 IO 和峰值显存；不是稀疏或低秩近似。GPU 算力高但 HBM 往返昂贵；分块 Q/K/V，维护行 max、归一化和与部分输出。计算复杂度主阶仍是二次。后端选择受 dtype、mask、head dim、GPU 和 dropout 影响。用 SDPA 对照输出，benchmark 前 warmup、同步 CUDA。

### Q78：Online Softmax 如何合并两个分块？

每块保存最大值 `m` 和指数和 `l`。合并时 `m=max(m1,m2)`，`l=exp(m1-m)l1+exp(m2-m)l2`；加权输出也按相同缩放合并。减最大值保证稳定；新块最大值更大时旧累积必须重标定。这个可结合律式状态让行 softmax 不用一次看到所有 key。输出分子 `o=Σexp(s-m)v` 同步按 `exp(m_old-m_new)` 缩放，最终除以 `l`。

### Q79：稀疏 MoE 的路由和前向流程是什么？

router 为每 token 产生 expert 分数，选 top-k，把 token dispatch 到相应 experts，expert FFN 计算后按 gate 权重 combine。总参数大但每 token 激活少数专家。训练要处理 capacity、负载均衡、expert parallel All-to-All、路由精度与 dropped tokens；推理还受 expert 权重加载和 batch 路由分散影响。`y=Σ_{i∈TopK}p_iE_i(x)`；记录每 expert token count、overflow 和 entropy。

### Q80：MoE 如何避免负载不均与 expert collapse？

监控每 expert token 比例、capacity overflow 和路由概率；可用 load-balancing auxiliary loss、capacity factor、路由噪声，或 DeepSeek-V3 报告的 expert-wise dynamic bias 方案。均衡不能牺牲主任务梯度而不评测。aux loss 同时鼓励重要性/负载均衡但系数过大会干扰表示；动态 bias 用路由统计调选择，不直接加入主 loss。分布式上最慢 expert 决定尾部。画 expert utilization 的 max/min、CV、熵与 All-to-All 时间。

### Q81：Dense 与 MoE 怎样公平比较？

先固定比较预算：训练 FLOPs、active parameters、总参数、token 数、质量或部署成本之一，再报告其他维度。MoE 每 token 激活少，但总权重、通信和服务内存仍大。训练要含 router/All-to-All，推理要说明 batch、并发和 expert placement；小模型/慢互联可能 dense 更高效。同时列 total params、active params/token、measured FLOPs、HBM、tokens/s。

### Q82：Scaling Law 能指导什么，不能保证什么？

它用经验幂律拟合 loss 随参数、数据和计算的变化，帮助在固定 compute 下选模型/数据配比；只在相近数据、架构和规模区间可信，不能保证下游能力或无限外推。区分 parameter scaling、data scaling、compute-optimal scaling；数据质量、去重、MoE active compute、长上下文和 post-training 会改变结论。可写 `L≈L∞+aN^{-α}+bD^{-β}`，用 log-log 拟合并留出规模验证。

### Q83：如何比较 Qwen3、DeepSeek-V3 和 LLaMA？

先明确 checkpoint/论文版本，再按 dense/MoE、attention/KV、norm/FFN、上下文、tokenizer、训练阶段和许可比较。不能用一个家族标签概括所有版本。可举例：DeepSeek-V3 报告 MLA、DeepSeekMoE、MTP；Qwen3 初版包含 dense/MoE 和 thinking/non-thinking 设计；LLaMA 1 论文是 dense decoder、RMSNorm/SwiGLU/RoPE。然后回到实际 `config.json`。现场列 `layers/hidden/heads/kv_heads/experts/top_k/vocab/max_pos`。

### Q84：LayerNorm、RMSNorm、BatchNorm 为什么使用场景不同？

LayerNorm 对单样本 token 的 hidden 维做均值/方差归一化；RMSNorm 只按均方根缩放；BatchNorm 用 batch 统计并维护运行均值。变长序列和小/动态 batch 的 LLM 更适合 LN/RMSNorm。说明 Pre-Norm 的位置、训练/推理统计是否一致；RMSNorm 少了中心化但不等于永远更好。`RMSNorm(x)=g⊙x/sqrt(mean(x²)+eps)`。

### 13.4 微调、RLHF 与推理对齐（10 题）

### Q85：Prompt Tuning、Prefix Tuning、Adapter 与 LoRA 如何选？

Prompt Tuning 学输入虚拟 token；Prefix Tuning 给各层 attention 学 K/V prefix；Adapter 插入瓶颈模块；LoRA 学线性层低秩增量。按任务容量、可训练参数、推理延迟、能否合并和多租户切换选。小模型上 soft prompt 容量可能不足；prefix 占 KV；adapter 增串行算子；LoRA 可 merge 但动态多 adapter 需服务支持。公平比较保持数据/steps/有效参数接近。报告 trainable params、额外 tokens/KV、P95 latency。

### Q86：AdaLoRA 相比固定 rank LoRA 改了什么？

固定 LoRA 给每层相同/手设 rank；AdaLoRA 在训练中根据重要性对低秩组件评分、剪枝和重分配，在总参数预算下把容量给更重要模块。说明预算调度、重要性估计和正交/稳定约束；优势依赖任务，训练和超参更复杂，不能只比较最终 rank。记录每层最终 rank 分布、可训练参数和验证指标。

### Q87：知识蒸馏有哪些形式，temperature 为什么常伴随 `T²`？

有 logits、feature、sequence-level 和 on-policy distillation。温度让类别分布更软；softmax 梯度随 `1/T` 缩放，KL 两侧共同影响常使梯度约缩小 `1/T²`，所以乘 `T²` 保持量级。学生 loss 常混 hard CE 与 soft KL；教师生成数据要去污染、验正确性；on-policy 让教师处理学生当前轨迹，缓解离线覆盖不足。`L=α CE+(1-α)T² KL(q_T||p_T)`。

### Q88：Reward Model 如何训练与验证？

给同一 prompt 的 chosen/rejected 评分，最小化 `-log σ(r_chosen-r_rejected)`。验证不只看 pair accuracy，还看人工一致性、长度/格式偏置、校准和分布外样本。数据需控制位置、长度和重复；reward 只在同一 RM 内相对有意义；高 pair accuracy 仍可能被 policy 利用产生 reward hacking。对交换顺序、截断、无关长答案做反事实测试。

### Q89：GRPO、PPO、DPO 的目标和数据要求如何比较？

DPO 用离线成对偏好和 reference 直接优化 log-ratio；PPO 用 on-policy rollout、reward/advantage、value baseline 和 clip；GRPO 对同 prompt 的一组 rollout 做组内相对 advantage，通常省去独立 value model。DPO 系统简单但受离线覆盖限制；PPO 灵活但组件多；GRPO 适合可批量验证的推理任务，但组内零方差、采样成本和 reward hacking 是风险。三者都需说明 KL/reference。写 DPO sigmoid 目标、PPO ratio clip、GRPO `A_i=(r_i-mean)/std`。

### Q90：GSPO 与 DAPO 分别解决什么，和 GRPO 有何关系？

GSPO 使用序列级 importance ratio、clipping 和优化，目标之一是提高长序列/MoE RL 稳定性；DAPO 是从 GRPO 出发的系统配方，加入 Clip-Higher、动态采样、token-level loss 与 overlong reward shaping 等。比较优化粒度、样本过滤、长度处理和论文实现；它们不是同一算法的别名。前沿结果必须限定 arXiv 版本、基座和任务。说明 token ratio 乘积易数值/方差问题，序列 likelihood 通常在 log 空间聚合。

### Q91：为什么 policy gradient 需要 baseline/GAE？

直接用 return 的 REINFORCE 无偏但方差大；减去与动作无关的 baseline 不改变期望梯度却降方差。GAE 用 `γλ` 加权多步 TD residual，在偏差与方差间折中。写 advantage `A=Q-V`；`λ→1` 更接近 Monte Carlo、方差高，`λ→0` 更依赖 value 估计、偏差可能高。LLM 序列长、终局 reward 稀疏，使信用分配更难。`δ_t=r_t+γV(s_{t+1})-V(s_t)`，`A_t=Σ(γλ)^lδ_{t+l}`。

### Q92：RLAIF 的收益和风险是什么？

RLAIF 用模型依据 rubric/原则产生偏好或评分，扩展快、成本低；风险是 judge 偏差、自偏好、位置/长度偏差、错误理由和同源模型相关错误。使用清晰 constitution/rubric、答案顺序交换、多 judge 或规则验证；抽样人工标注做一致性与阈值校准，保留 judge 版本。高风险领域不能完全替代人。报告与人工 Cohen's κ/相关性、swap consistency、分桶准确率。

### Q93：结果奖励、过程奖励、token/sequence reward 如何做信用分配？

结果奖励便宜可靠但稀疏；过程奖励更密集却需要可信步骤验证。sequence reward 给整条回答一个分数，token-level loss 仍需把 advantage 分配到 token；错误分配会鼓励冗长或投机。可用 value/GAE、reward-to-go、过程 RM、规则 verifier 或组内相对优势。Agent 工具轨迹还要区分决定、工具执行与环境随机性。按 response token mask 聚合，分别监控长度与 reward 的相关性。

### Q94：SFT 为什么常用 assistant-only loss？数据越多越好吗？

assistant-only mask 只让模型拟合理想回复，避免把用户输入也当目标；但预训练/继续预训练或特定模板可用不同 mask。数据不是越多越好，重复、错误、模板单一和冲突样本会伤害模型。核对 chat template 边界、EOS、packing 后样本隔离；按来源/任务/长度分桶，去重与质量评分，建立通用能力回归。比较按样本 epoch 不如按有效训练 token/steps 清楚。打印一条 token—label 对齐，非目标位置应为 `-100`。

### 13.5 推理、量化与分布式（8 题）

### Q95：Prefill 与 Decode 的计算特征为什么不同？

Prefill 并行处理整段 prompt，矩阵较大、常偏 compute-bound，决定首 token 前的主要模型计算；Decode 每步为每请求生成一个 token，反复读取权重和 KV Cache，常偏 memory-bandwidth-bound。长 prompt 会拉高 prefill 和排队；decode batching 可提高吞吐但影响尾延迟。大并发或超长 context 下 KV 容量和调度成为约束。分别按 input/output length 分桶报告 prefill time、TTFT、TPOT。

### Q96：TTFT、TPOT、吞吐和端到端延迟如何一起评测？

TTFT 是请求至首 token，TPOT/ITL 是首 token 后输出 token 间隔，吞吐是单位时间完成的 tokens/requests，端到端是最后 token 返回。必须给 P50/P95/P99、并发和长度分布。区分 server/client observed、input/output tokens/s、goodput；做 open-loop 到达率压测，观察饱和点和排队，而不是只做单请求 benchmark。`E2E≈TTFT+(n_out-1)×TPOT` 只是平均近似。

### Q97：PagedAttention 解决什么问题？

把每请求 KV Cache 分为固定逻辑块，映射到可非连续物理块，按需分配并减少预留、内部/外部碎片；还能支持共享相同前缀块。attention 数学不变。传统连续预分配需按最大长度留空间，请求长度动态会浪费；分页让调度器复制/共享块表。块太小元数据/调度开销大，太大尾块浪费多。监控 GPU KV cache usage、block allocation、eviction/preemption。

### Q98：Continuous batching 与 prefix cache 怎样提高吞吐，有什么风险？

continuous batching 在 decode step 边界动态加入/移除请求，减少 GPU 空槽；prefix cache 复用完全相同 token 前缀的 KV，跳过重复 prefill。风险是排队公平、cache 污染/泄漏、错误 key 和尾延迟。prefix key 至少关联 token ids、模型/adapter、位置/多模态输入和租户安全域；动态 batching 要设置 token budget、优先级和 preemption。分别测 cache hit rate、saved prefill tokens、queue time、preemption。

### Q99：GPTQ、AWQ、GGUF 有什么区别？量化粒度如何影响结果？

GPTQ 是校准后权重量化方法，利用近似二阶信息逐层减小重构误差；AWQ 用激活统计保护显著权重；GGUF 是 llama.cpp 生态的模型容器/量化格式，不是训练算法。per-channel/group 比 per-tensor 细，通常误差小但元数据和 kernel 更复杂。说清 W/A/KV、位宽、group size、对称/非对称、校准集和硬件后端；量化质量与速度必须分别测。`x_q=clamp(round(x/s)+z)`，反量化 `s(x_q-z)`。

### Q100：怎样估算模型训练和推理显存？

推理是权重、KV、激活/workspace；训练还含梯度、优化器 moments、可能的 FP32 master weights、保存激活、通信 bucket 与碎片。先逐项写 bytes，再用实测校正。KV `2BLTHkvDh×bytes`；AdamW 全参至少考虑两份 moments。LoRA 只减少可训练参数相关状态，不消除基座权重/激活；ZeRO/FSDP 只对分片项近似除 world size。同时记录 `max_memory_allocated` 与 `reserved`，后者含 allocator 缓存。

### Q101：FSDP 与 DeepSpeed ZeRO 怎么选？

二者都能分片训练状态；FSDP 是 PyTorch 原生分片/包装路径，便于原生生态组合，ZeRO 属于 DeepSpeed engine，offload 和训练系统能力成熟。选型看团队栈、模型支持、checkpoint、offload、网络和实测吞吐，不只看“等价 stage”。比较 wrap 粒度、参数 all-gather 时机、backward reduce-scatter、prefetch、mixed precision 和 state-dict；ZeRO-1/2/3 分片 optimizer/再梯度/再参数。保持 global batch/sequence/精度相同，测 peak HBM、tokens/s、恢复时间。

### Q102：All-Reduce、All-Gather、Reduce-Scatter、All-to-All 各用在哪里？

DDP 用 All-Reduce 同步梯度；分片训练常用 Reduce-Scatter 梯度、All-Gather 参数；TP 会按切法用 All-Reduce/All-Gather；MoE EP 用 All-to-All 分发 token。ring All-Reduce 每 rank 大约通信 `2(N-1)/N` 个 tensor；小消息受 latency，大消息受 bandwidth。跨节点拓扑、bucket 大小和计算通信重叠决定 scaling。先测 NCCL microbenchmark，再看 profiler 中 collective 与 kernel overlap。

### 13.6 Embedding 与高级 RAG（10 题）

### Q103：如何为 RAG 选择和评测 Embedding 模型？

先用真实 query—evidence 集，比较 Recall@K、MRR/nDCG、维度、最大长度、语言/领域、延迟和存储；通用榜单只做候选筛选。必须使用模型要求的 query/document 前缀和归一化方式。按专名、短问、长问、多语言、表格等分桶；排除数据污染；控制同一 ANN 参数和 chunk corpus，再做端到端评测。cosine 在向量 L2 归一化后等价于 dot product 排序。

### Q104：Embedding 如何训练？hard negative 怎么构造？

双塔编码 query/positive，用 in-batch negatives 的 InfoNCE 拉近正例、推远负例。hard negatives 从 BM25/dense 高排名但不含答案的文档中选，并过滤实际相关的假负例。温度控制分布尖锐度；batch 越大 negatives 多但跨设备 gather 成本高；可做多正例、去偏和 hard-negative curriculum。`-log exp(sim(q,d+)/τ)/Σ_j exp(sim(q,d_j)/τ)`。

### Q105：解释 BM25；为什么它仍适合 LLM RAG？

BM25 用 IDF、词频饱和和文档长度归一化评分。它对产品名、错误码、编号和精确术语强，且无需训练，所以常与 dense 互补。`k1` 控制 TF 饱和，`b` 控制长度归一化；中文需合适分词/字符策略。混合可用 RRF，避免直接相加不可比的分数。写 `IDF(q)·tf(k1+1)/(tf+k1(1-b+b|D|/avgdl))`。

### Q106：HNSW 的 `M`、`efConstruction`、`efSearch` 分别影响什么？

`M` 控制每节点连接度和内存/图质量；`efConstruction` 控制建库搜索宽度，越大建库慢但图通常好；`efSearch` 控制查询候选宽度，越大 recall 高但 latency 增。HNSW 用分层小世界图从上层粗定位到底层精搜；删除/更新和过滤会影响连通性与性能。调参必须在真实 corpus/并发上测 recall-latency。用 exact brute-force top-k 作 recall gold。

### Q107：什么是 Lost in the Middle，怎样验证和缓解？

模型对长 context 中间位置的关键信息利用可能弱于开头/结尾。用同一证据在不同位置的受控实验验证；缓解包括提高 rerank/去噪、缩短 context、将关键证据前置并保留引用。不要把所有长文失败都归因于位置；还可能是冲突、token 截断或问题表达。对照应固定内容和 token 数，仅交换位置。画 position bucket vs exact match/faithfulness。

### Q108：多跳 RAG 如何分解问题并避免错误级联？

先识别需要哪些中间实体/事实，生成可检索子问题，逐跳检索并保存每跳 evidence，再合成答案；每跳做置信度和来源检查，证据不足时停止/澄清。query decomposition 可并行或依赖执行；中间答案不要无引用地当事实传给下一跳。设置最大跳数、去重和回溯，使用 gold evidence chain 分层评测。记录 hop Recall、链完整率、最终正确率和额外 latency。

### Q109：GraphRAG 适合什么问题，为什么不能替代普通 RAG？

GraphRAG 抽取实体/关系并构建社区或摘要，适合跨文档关系、多跳和全局主题问题；普通局部事实查询用向量/BM25 更简单便宜。图抽取错误、更新成本和权限传播是主要风险。讲清实体消歧、边来源、community summary、local/global search；所有边要回溯原文。做路由而非全量强制走图。评测 relation extraction、evidence coverage、answer 与 build/update cost。

### Q110：RAG 如何做增量更新、删除和时间衰减？

文档有稳定 ID、版本、content hash、有效时间和权限；变化时只重建受影响 chunks，upsert 新版本、原子切换，再删除旧向量并失效缓存。时间衰减只用于确实随时间变旧的内容。维护 doc→chunk→embedding lineage；删除要覆盖向量、关键词、图、缓存和副本；政策冲突按有效期/权威级别，不只是 newest wins。可用 `score'=score×exp(-λΔt)`，但 λ 要在时间敏感测试集调。

### Q111：表格、图片、OCR 和噪声文档如何进入 RAG？

保留页码、标题层级、表格行列、bbox、图片区域和 OCR 置信度；同时生成适合检索的结构化文本，并保留原始页面/区域供 VLM 核验与引用。低置信/重复/页眉页脚要过滤或降权。表格按行切会丢表头，需携带表名与列名；图片 caption 不足以回答细节，要视觉 embedding/区域裁剪；解析失败单独评测。schema 至少有 `doc_id,page,bbox,modality,content,version,confidence`。

### Q112：什么时候用 RAG、搜索、长上下文或微调？

外部知识频繁更新且需引用用 RAG；开放网络时效信息用搜索并校验来源；少量静态文档可直接长上下文但受成本/位置影响；行为、风格和稳定格式用 SFT。它们可组合。先按知识 freshness、私有性、规模、可引用、延迟和离线要求定约束。检索失败不应用微调硬记补救；微调也不能可靠更新事实库。用 decision table 和同一测试集比较质量—延迟—成本。

### 13.7 Agent、MCP、Memory、安全与评测（12 题）

### Q113：CoT、ToT、Reflexion、ReWOO 有何区别？

CoT 生成单条中间步骤；ToT 搜索多条 thought 分支并评分；Reflexion 根据失败反馈总结再尝试；ReWOO 先规划带依赖的工具步骤，再执行和汇总。成本、可控性与适用任务不同。简单任务 CoT/直接答；有可评分分支用 ToT；环境给真实反馈时 Reflexion 才可靠；工具依赖稳定时 ReWOO 可减少 LLM 往返。生产记录简短依据和工具轨迹，不依赖暴露完整私有推理。统一 max steps/token/tool budget 后比较 success/cost。

### Q114：什么是 12-Factor Agents？你实际采用了哪些原则？

它是 Agent 工程原则集合：自己控制 prompt/context/control flow，把 tool call 当结构化控制输出，统一执行/业务状态，支持 pause/resume、人类介入、紧凑错误、小型专职 agent 和可重放 reducer。不是协议或框架。选项目中真实的三项举证，例如 state event log 可重放、write tool 人工确认、错误压缩后重试；解释未采用多 agent 的原因。给一次 run 的 state transitions 与 event IDs。

### Q115：MCP 的 host、client、server 如何协作？

host 管理用户体验、安全和多个 client；每个 client 与一个 server 会话通信；server 暴露 tools/resources/prompts 等能力，使用 JSON-RPC 消息并先做版本/能力协商。tools 是可调用动作，resources 是可读取上下文，prompts 是模板能力；传输和生命周期按规范实现。MCP 只标准化连接，不自动完成授权、沙箱、审计或幂等。先 `initialize`/协商，再列能力、调用、处理 error/cancel；固定规范版本。

### Q116：MCP 与普通 function calling 有什么区别？

function calling 是模型按给定 schema 产出工具名/参数的能力；MCP 定义 host/client/server 的发现、能力、消息和生命周期。MCP tool 最终仍可通过模型 function calling 选择，两者在不同层。少量本地固定工具直接 function calling 足够；跨应用复用/动态发现/统一资源时 MCP 有价值。无论哪种都需业务授权、参数校验、超时和结果净化。画 `model→tool request→policy gate→MCP client→server`。

### Q117：A2A 与 MCP 的定位是什么？能否一起用？

MCP 更偏 host/agent 接入工具和上下文；A2A 面向独立 agents 发现能力、委派任务、交换状态和产物。一个 agent 可用 A2A 接任务，再通过 MCP 使用工具，因此可以组合。跨组织 agent 要处理身份、授权、任务状态、超时、取消、产物引用和不可信消息；采用前固定 A2A v0.3.0 等具体规范版本。任务状态机至少含 submitted/working/completed/failed/canceled。

### Q118：如何让 Agent 稳定输出结构化数据？

给 JSON Schema/Pydantic 契约，能用受限解码就约束语法；生成后 parse、schema 校验、业务校验。把短结构化错误返回模型重试，超过预算转 fallback/人工。语法合法不代表 ID 存在、金额合法或权限允许；版本化 schema，区分可重试与不可重试错误；工具结果也视为不可信输入。错误形如 `{"code":"INVALID_DATE","field":"start","retryable":true}`。

### Q119：Context 超长时如何压缩而不丢关键状态？

把系统约束、当前目标、未完成事项、关键实体和工具结果引用设为不可丢；其余按 recency/relevance 裁剪或摘要，原始事件存外部日志，可按 ID 取回。分开 message history、execution state、retrieved evidence；摘要带来源和版本，重要数字/否定约束结构化保存。对压缩前后在长任务集做等价性回归。token budget 分配给 rules/state/evidence/recent messages，并设溢出策略。

### Q120：Agent 工具沙箱和权限如何设计？

每个工具只获必需文件、网络、命令和凭证；读写分离，危险操作 allowlist/人工确认，设置超时、资源限额、幂等键和审计。外部文本永远不能直接升级权限。按 tenant/run 隔离 workspace 和 secret；网络出站白名单；参数规范化防路径穿越/命令注入；工具结果长度和内容净化；失败后回滚或补偿。policy gate 输入 user/tenant/tool/args/risk，输出 allow/deny/confirm。

### Q121：并发 Agent 如何防止状态串线和重复副作用？

所有状态带 `tenant_id/session_id/run_id`；避免共享可变全局对象，更新用版本号/乐观锁或事务，工具调用带唯一 call/idempotency key，结果按 call id 合并。并行工具不能依赖未完成结果；定义 join/reducer 和冲突策略；消息至少一次投递会重复，因此写工具必须幂等；取消后迟到结果不能覆盖新状态。CAS：仅当 `state.version==expected` 才提交，失败则重读/重算。

### Q122：Task、Trial、Grader、Harness 分别是什么？

Task 定义输入、初始环境和成功标准；Trial 是一个 agent 配置在 task 上的一次运行；Grader 根据结果/轨迹打分；Harness 负责建隔离环境、执行 trials、采集日志和聚合。task 数据需版本化，trial 独立且可重放，grader 可为代码/model/human 并有校准；harness 固定依赖、时间、随机性和清理。结果主键 `(task_version,agent_version,trial_seed)`。

### Q123：Outcome 与 Transcript 为什么要分开评；用户模拟器如何校准？

Outcome 看任务最终是否完成；Transcript 看是否违规、绕路、调用错工具或泄漏信息。用户模拟器提供隐藏目标和多轮回应，但需与真实对话在行为/难度上校准，不能让它既出题又无约束评分。结果正确也可能通过危险操作得到；过程合规也可能没完成。模拟器固定 persona、知识边界和随机种子，用人工样本比较澄清率/对话长度/成功率。组合评分设 hard safety gate，再加 outcome/efficiency，而非加权掩盖违规。

### Q124：`pass@k` 与 `pass^k` 有什么区别？模型 Grader 如何校准？

pass@k 是 k 次中至少一次成功，反映探索上限；pass^k 是 k 次全部成功，反映稳定性。若独立成功率 p，近似为 `1-(1-p)^k` 与 `p^k`。模型 grader 要对人工 gold 做一致性、位置交换和分桶校准。实际 samples 非独立时不能只套公式；按 task 多 trial 再宏平均，报 CI。grader 检查长度、风格、自偏好和 prompt injection，保留 rubric/version。同时报告 pass@1、pass@k、pass^k、成本；bootstrap task-level CI。

### 13.8 多模态与论文解读（6 题）

### Q125：CLIP 的 InfoNCE 如何对齐图像和文本？

图像和文本 encoder 输出归一化向量，计算 batch 内相似度矩阵；配对图文为正例，其他为负例，分别做 image-to-text 与 text-to-image cross-entropy，温度控制分布尖锐度。大 batch 提供更多 negatives，但会有假负例；CLIP 学检索/对齐空间，不直接生成。zero-shot 分类把类别写成 text prompts 比相似度。`L=(CE(S/τ,diag)+CE(S^T/τ,diag))/2`。

### Q126：多模态 early、late、cross-attention 融合如何比较？

early/token fusion 早期拼接模态 token，交互充分但 context/计算大；late fusion 独立编码后合并，适合检索和扩展；cross-attention 让一侧按需读取另一侧，在交互强度和成本间折中。projector+LLM 属常见 token 接入；双塔适合离线预计算；cross-attention 需处理位置、mask 和缓存。按任务是否需要细粒度 grounding 选择。画 tensor shapes 和哪些参数冻结/训练。

### Q127：视觉 Grounding 是什么，如何训练和评测？

Grounding 把文本实体/短语绑定到图像中的 bbox、point 或 mask。训练数据要有区域标注和统一坐标协议；评测用 IoU/point accuracy/phrase recall，并与答案正确率分开。坐标可离散成 token 或由检测头预测；resize/padding 后要正确映射回原图。通过换图、遮挡目标和相似干扰物做反事实。bbox IoU=`intersection/union`，明确 IoU threshold。

### Q128：高分辨率图像为何困难？dynamic resolution/tiling 有什么权衡？

固定缩小会丢小字细节，直接高分辨率使 patch/token 数和 attention 成本激增。tiling/dynamic resolution 保留局部细节，但增加 token、负载不均并可能丢全局关系。常用缩略图+局部块、按长宽比选网格、token pooling/pruning；必须携带 tile 坐标、去重重叠区域，并控制最大视觉 token。ViT patch 数约 `(H/P)×(W/P)`，分辨率翻倍 token 约四倍。

### Q129：视频多模态如何建模时间信息？

先做均匀/事件感知帧采样，再用时间位置编码、时序模块或跨帧 attention 聚合。静态图模型逐帧看会丢动作顺序、持续时间和跨帧实体一致性。均匀采样可漏短事件，密集采样耗 token；可粗到细二阶段选帧。训练/评测覆盖顺序、计数、状态变化和长视频检索。固定总视觉 token，对比帧数×每帧分辨率配置。

### Q130：如何设计多模态 RAG，并防止模型只读 OCR 文本？

同时索引页面文本、表格结构、图像/区域 embedding 与元数据；query 路由或混合召回，rerank 后把原页面区域和文本证据交给 VLM。用遮图、换图、OCR 冲突测试确认视觉被使用。保留 doc/page/bbox/版本，回答引用到区域；分别评跨模态 Recall、OCR/表格答案、grounding 与 faithfulness。恶意 OCR 指令作为不可信数据隔离。做 text-only、image-only、multimodal 三基线和 RRF/跨编码 rerank。

## 14. 手撕练习与项目追问模板

### 14.1 二十道手撕练习

每题要求：先口述复杂度与边界，20–30 分钟独立实现，至少写 3 个测试；框架题再与官方实现对齐。

| # | 练习 | 必测边界/考点 |
|---:|---|---|
| 1 | Scaled Dot-Product Attention | causal/padding mask、全 mask、dtype |
| 2 | Multi-Head Attention | reshape/transpose/contiguous、输出 shape |
| 3 | MQA/GQA | KV repeat/group、cache 大小 |
| 4 | Online Softmax | 分块合并、极大 logits |
| 5 | LayerNorm/RMSNorm | 维度、eps、与 PyTorch 对齐 |
| 6 | LoRA Linear | 初始化、scale、merge/unmerge 等价 |
| 7 | Assistant-only CE | shift、`-100` mask、有效 token 平均 |
| 8 | 梯度累积训练循环 | 尾 batch、AMP unscale/clip、zero grad |
| 9 | Top-k sampling | k 边界、renormalize、seed |
| 10 | Top-p sampling | 排序、累计阈值、至少保留一个 |
| 11 | BM25 | TF 饱和、文档长度、OOV |
| 12 | RRF | 缺失文档、并列排名、多个 retriever |
| 13 | cosine Top-K | 归一化、batch、tie |
| 14 | LRU Cache | 哈希表+双向链表、容量 0/1 |
| 15 | 最长无重复子串 | 滑窗不回退、Unicode/空串 |
| 16 | 数组 Top-K | heap 与 quickselect 复杂度 |
| 17 | 合并区间 | 排序、相接区间、空输入 |
| 18 | 反转/判环链表 | 空链、单节点、Floyd |
| 19 | 二叉树 BFS/DFS | 递归深度、层序、空树 |
| 20 | 有依赖工具的拓扑排序 | 环检测、稳定顺序、缺失节点 |

不要背一份“漂亮代码”；练习现场先澄清输入 shape、mask 语义、是否允许原地修改和错误策略。项目代码中若调用框架算子，也要能写出最小正确版本和单元测试。

### 14.2 项目追问九宫格

对简历每个项目准备下表，所有 `[占位符]` 只填真实信息：

| 追问轴 | 必须讲清 | 可验证材料 |
|---|---|---|
| 数据来源 | 来源、许可、规模、去重、划分、泄漏 | data card、hash、样本审计 |
| 模型选择 | 任务约束、候选、为何不选替代 | model card、基线表 |
| 参数依据 | LR/rank/chunk/top-k/阈值如何选 | sweep 配置、曲线 |
| 资源规模 | GPU/显存/时长/token/并发口径 | env、账单或监控日志 |
| 基线 | 最简单可用方案与强基线 | 固定 eval 输出 |
| 消融 | 一次只变一个关键因素 | experiment matrix |
| Bad case | 分类、根因、修复与未解局限 | samples、issue、回归测试 |
| 上线迭代 | API、SLO、安全、监控、回滚 | load test、dashboard、runbook |
| 团队分工 | 你亲自负责什么，接口如何协作 | commit/PR、设计记录 |

统一 2–3 分钟表达：

```text
目标与约束 → 最小基线 → 我负责的关键实现 → 参数/方案依据
→ 固定测试集上的真实结果 → 最大 bad case 与定位证据
→ 改进后的对照 → 当前局限和下一步
```

若没做过某项，回答“当前未实现；我会先用 `[验证方法]` 判断是否值得做”，不要把计划说成经历。

## 15. 面试前最终检查

- [ ] 10 分钟内手写 Attention，形状和 mask 正确。
- [ ] 手算一个 LoRA 层参数量，写出 DPO log-ratio。
- [ ] 能解释一条真实训练日志中的 lr、loss、grad norm、tokens/s。
- [ ] 准备至少一个 OOM/NaN/低吞吐的真实排障故事。
- [ ] RAG 项目能分别报告 retrieval 与 generation 指标。
- [ ] Agent 项目能展示失败轨迹、终止条件和工具安全。
- [ ] 每个简历数字都能找到日志或结果文件。
- [ ] 不熟悉的前沿缩写先确认论文语境，不强答。

## 参考资料

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers](https://huggingface.co/docs/transformers/index)、[PEFT](https://huggingface.co/docs/peft/index)、[TRL](https://huggingface.co/docs/trl/index)
- [DeepSpeed ZeRO](https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/zero.md)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [LangGraph](https://github.com/langchain-ai/langgraph)、[InternVL](https://github.com/OpenGVLab/InternVL)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[FlashAttention](https://arxiv.org/abs/2205.14135)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)与[官方仓库](https://github.com/QwenLM/Qwen3)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)
- [vLLM](https://docs.vllm.ai/)、[MCP 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic)、[A2A v0.3.0](https://a2a-protocol.org/v0.3.0/specification/)

> 访问日期：2026-07-14。涉及框架 API 的题，应结合自己项目实际锁定的版本回答。本地 GitHub 面经只用于发现考点与追问；自动归并频次不是招聘概率，技术答案已按上述 primary sources 纠错。
