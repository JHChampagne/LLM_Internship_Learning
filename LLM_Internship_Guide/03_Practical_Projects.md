# 大模型算法工程实战项目手册

> 目标：用 8–12 周完成 5 个可复现、可评测、可面试深挖的小项目。更新基线：2026-07-14。  
> 配套：[技术学习指南](./01_Technical_Guide.md)｜[面试题库](./02_Interview_QA.md)

## 目录与学习目标

1. 项目完成等级、公共环境与实验规范
2. 从零实现并训练 Mini Transformer
3. Qwen 领域模型 SFT、QLoRA 与 DPO
4. 带分层评测的企业知识库 RAG
5. LangGraph 多工具研究 Agent
6. InternVL/Qwen-VL 多模态问答微调
7. DeepSpeed ZeRO 与 Megatron 两卡 TP 附加实验
8. 8–12 周排期、招聘要求覆盖矩阵、简历写作和发布检查

**先修知识**：会使用 Python、Git、Linux 和 PyTorch 基础训练循环；开始项目 2–5 前应读完技术指南对应章节。  
**学习目标**：每个项目至少做到 L2，重点项目做到 L3/L4；能展示配置、日志、指标、错误样本和复现步骤，而不只是演示成功页面。

## 0. 先读：什么才算“做过项目”

克隆仓库、跑通作者 Demo 只能算 **L1**，不能直接写成“独立设计并实现”。本手册统一使用以下完成等级：

| 等级 | 标准 | 可否写简历 |
|---|---|---|
| L1 复现 | 在自己的环境成功运行官方基线 | 可写“复现”，不写“设计” |
| L2 迁移 | 换成自己的数据，跑完训练/索引/评测 | 可写“基于…实现” |
| L3 实验 | 提出改动，完成公平对照和错误分析 | 可写“优化”，必须有数字 |
| L4 交付 | API/Demo、README、配置、日志、结果图都可复现 | 可作为主项目展开 |

简历中的每个数字必须能追溯到 `metrics.json` 或原始日志。模板中的 `[占位符]` 在未完成前不得删除或编造。

### 0.1 公共环境与版本策略

推荐 Linux、Python 3.11+、NVIDIA GPU。先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装匹配驱动的版本，再安装项目依赖。框架更新快，不跨项目共用一个“万能环境”。

```bash
mkdir -p llm-portfolio && cd llm-portfolio
git init
mkdir -p runs reports assets
nvidia-smi | tee reports/gpu.txt
python --version | tee reports/python.txt
```

对每个外部项目记录 commit，而不是只写 `main`：

```bash
git rev-parse HEAD | tee USED_COMMIT.txt
python -m pip freeze > requirements-lock.txt
```

### 0.2 云 GPU 资源建议

| 任务 | 最低可用 | 推荐 | 降级方案 |
|---|---|---|---|
| Mini Transformer | 1×8GB | 1×16–24GB | 缩短序列/模型 |
| Qwen 4B QLoRA | 1×16–24GB，依配置而定 | 1×24–48GB 或 2×24GB | 改 0.6B/1.7B |
| RAG/Agent | CPU+API 可跑 | 1×16–24GB 本地模型 | 使用兼容 API |
| VLM LoRA | 1×24GB 小模型/小分辨率 | 2–4×40/80GB | 只训 projector/更小 VLM |
| Megatron TP | 2 卡 | 同节点 2–4 卡高速互联 | 只跑 mock/minimal loop |

显存受版本、序列、图片分辨率、batch、optimizer 和 checkpointing 影响，表中不是承诺值。先跑 10 step smoke test，再租长时实例。

### 0.3 每个项目统一目录

```text
project_x/
├── README.md                 # 一条命令复现、架构、结果、局限
├── configs/                  # 所有实验配置
├── src/                      # 自己写的核心代码
├── data/README.md            # 来源、许可、schema、清洗、划分
├── tests/                    # 单测和最小集成测试
├── runs/                     # 原始日志，不提交大 checkpoint
├── reports/
│   ├── metrics.json
│   ├── error_analysis.md
│   └── figures/
└── demo/                     # API/UI/截图或录屏
```

### 0.4 实验报告最小模板

```markdown
# 实验名称
- 假设：为什么这个改动可能有效？
- 固定项：数据划分、模型、训练预算、解码参数、评测脚本。
- 自变量：只改变什么？
- 结果：均值、样本数/种子、显存、耗时。
- 错误分析：至少 20 个失败样本分桶。
- 结论：假设是否成立、局限、下一步。
```

---

## 项目 1：从零实现并训练 Mini Transformer

### 1.1 项目目标

不依赖高层 Trainer，实现一个 Decoder-only Causal LM，证明你理解 Tokenizer、Attention、Block、loss shift、训练循环、生成、AMP、梯度累积和 DDP。最终产出 loss 曲线、速度/显存曲线和至少两项消融。

### 1.2 技术栈与目录

```text
mini_transformer/
├── configs/base.yaml
├── src/{data.py,model.py,train.py,generate.py}
├── tests/{test_attention.py,test_overfit.py}
└── reports/{benchmark.csv,error_analysis.md}
```

依赖：`torch`、`transformers`（只借 tokenizer）、`datasets`、`pyyaml`、`tensorboard`。可参考 [nanoGPT](https://github.com/karpathy/nanoGPT) 的训练组织，但核心层必须自己写并能解释。

### 1.3 数据与配置

从无版权争议的小型公开文本或自建文本开始。先用 1–10MB 验证，再扩到足以看到泛化的规模。保存 train/val 文件 hash，避免同段落跨集合。

```yaml
seed: 42
vocab_model: gpt2
context_length: 256
n_layers: 6
n_heads: 6
d_model: 384
dropout: 0.1
micro_batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 0.0003
weight_decay: 0.1
warmup_steps: 200
max_steps: 10000
precision: bf16
grad_clip: 1.0
```

### 1.4 必须自己实现的核心

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

训练循环必须包含：

- seed、train/eval、正确 label shift/mask。
- AdamW 参数分组和 warmup+cosine scheduler。
- BF16/FP16 autocast；FP16 时使用 GradScaler。
- gradient accumulation、clip、checkpoint/resume。
- 定期验证、tokens/s、峰值显存、grad norm 日志。

### 1.5 单元测试

1. **Causal 测试**：改变未来 token，不应改变过去位置输出。
2. **形状测试**：不同 batch/sequence 输出 `[B,T,V]`。
3. **Attention 对齐**：与 PyTorch SDPA 在同权重、无 dropout 时误差小于设定阈值。
4. **单 batch 过拟合**：几十步把一个 batch loss 显著降下来；失败说明实现或优化有问题。
5. **Resume 测试**：连续训练 N 步与 N/2 保存恢复后的 loss/参数在容差内一致。

```bash
pytest -q
torchrun --standalone --nproc-per-node=2 -m src.train --config configs/base.yaml
tensorboard --logdir runs --port 6006
```

### 1.6 实验矩阵

| 实验 | 自变量 | 固定项 | 记录 |
|---|---|---|---|
| Norm | Pre-Norm vs Post-Norm | 参数量、步数 | loss、grad norm、NaN |
| FFN | GELU vs SwiGLU | 近似参数量 | val loss、tokens/s |
| 长度 | 128/256/512/1024 | model、有效 tokens | 显存、耗时 |
| AMP | FP32/BF16/FP16 | batch | loss、速度、scaler skip |
| 扩展 | 1/2/4 GPU DDP | 每卡 batch或global batch需说明 | 吞吐、scaling efficiency |

DDP 两种口径分别报告：固定 per-GPU batch 看最大吞吐；固定 global batch 看纯扩展效率。不要混在同一结论。

### 1.7 成功验收与排错

- L1：100 step loss 可下降，生成接口可用。
- L2：训练/验证分离，能生成连贯局部模式。
- L3：完成至少两项公平消融和 DDP scaling。
- L4：单测、resume、报告、Demo 齐全。

常见故障：loss 不降先做单 batch 过拟合；生成乱码先检查 tokenizer、shift、EOS 和采样；OOM 区分参数/激活；多卡卡死先找最早异常 rank 和 collective 不一致。

### 1.8 高频扩展：GQA、FlashAttention、Mini-MoE 与数学测试

在基础模型收敛后创建独立分支，不要同时改注意力、FFN 和训练配置。

**A. MHA/MQA/GQA**

1. 把 `n_heads` 与 `n_kv_heads` 分离；K/V 为 `[B,Hkv,T,Dh]`，计算前按 group 映射到 query heads。
2. 依次跑 `Hkv=Hq`（MHA）、`Hkv=Hq/4`（GQA）、`Hkv=1`（MQA）。保持层数、`d_model`、训练 token 和解码配置一致。
3. 记录参数量、KV bytes、验证 loss、生成质量、prefill/TPOT。若训练参数量略有变化，明确列出，不声称是完全同容量。

**B. SDPA/FlashAttention benchmark**

用同一 Q/K/V、mask、dtype 比较手写 eager attention 与 `scaled_dot_product_attention`。每个 shape 先 warmup，计时前后 `torch.cuda.synchronize()`；从 PyTorch backend 日志确认实际 kernel，不能把“调用 SDPA”写成“必然使用 FlashAttention”。输出 `benchmark_attention.csv`：`B,T,H,Dh,dtype,backend,latency_ms,peak_mb,max_error`。

**C. 可选 Mini-MoE**

把一个 FFN 替换成 4 experts、top-2 router；记录每 expert token count、路由熵、capacity overflow 和总/激活参数。加入可开关的 load-balancing loss，对比无均衡项、不同系数。小规模实现可先单卡 dispatch，不伪装成 EP；多卡 All-to-All 留作进阶。

**D. 数学正确性测试**

```python
def test_softmax_ce_gradient():
    z = torch.randn(3, 7, requires_grad=True, dtype=torch.float64)
    y = torch.tensor([1, 2, 0])
    torch.nn.functional.cross_entropy(z, y, reduction="sum").backward()
    expected = torch.softmax(z.detach(), -1)
    expected[torch.arange(3), y] -= 1
    torch.testing.assert_close(z.grad, expected)
```

再增加：online softmax 分块合并、LoRA merge/unmerge、RMSNorm 与 PyTorch/参考实现、GQA cache shape、全 mask 行行为。CI 至少跑 CPU 单测；GPU benchmark 不设脆弱的绝对性能断言。

**真实面试追问**：为什么 FlashAttention 计算仍是二次？GQA 省的是哪部分显存？MHA→GQA 权重怎么初始化？MoE 为什么未必更快？你的结果是 kernel 还是算法导致？  
**必须保存的证据**：实现 commit、`pytest` 输出、backend 日志、benchmark 原始 CSV、显存曲线、GQA/MoE 消融表、一个失败实验及原因。

### 1.9 面试与简历

**60 秒讲法**：目标是理解底层而非追求大参数；说明模型规模、数据、亲手实现的层、一个消融和一个分布式瓶颈。

```text
- 使用 PyTorch 从零实现 [参数量] Decoder-only Transformer，包含 RoPE/RMSNorm/
  SwiGLU、AMP、梯度累积与 checkpoint；在 [GPU×数量] 上完成 DDP 扩展实验，
  吞吐由 [A] 提升至 [B] tokens/s，扩展效率 [C]%，并通过 causal、SDPA 对齐和
  resume 单元测试验证正确性。
```

---

## 项目 2：Qwen 领域模型 SFT、QLoRA 与 DPO

### 2.1 项目目标

用自己的领域数据完成 base 评测 → SFT/QLoRA → DPO → adapter/合并模型评测 → OpenAI-compatible API 部署。建议选择能客观判定的领域，如结构化信息抽取、课程答疑、命令解释；不要只做“风格聊天”。

训练入口采用 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)。其官方仓库在本手册访问时覆盖 Qwen3/Qwen3-VL、SFT、DPO、LoRA/QLoRA、DeepSpeed 和 vLLM/SGLang 部署。接口会更新，执行时固定 commit 并阅读该 commit 的 `examples/README.md` 与 `data/README.md`。

### 2.2 安装与 smoke test

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
git rev-parse HEAD | tee USED_COMMIT.txt
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[torch,metrics]"
llamafactory-cli version
# 先按官方当前示例跑 Qwen LoRA smoke test：
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

若 extras 名称随版本变化，以仓库 Installation 为准。不要把安装失败通过随意降级几十个包“修好”；新建环境并记录兼容矩阵。

### 2.3 数据设计

SFT 数据每行示例：

```json
{"messages":[{"role":"system","content":"你是课程知识助手。"},{"role":"user","content":"解释梯度累积。"},{"role":"assistant","content":"梯度累积是……"}],"meta":{"source":"manual","topic":"training"}}
```

DPO 数据需要同一 prompt 下的 chosen/rejected。实际字段必须与所固定 LLaMA-Factory commit 的数据说明一致：

```json
{"prompt":[{"role":"user","content":"解释梯度累积。"}],"chosen":[{"role":"assistant","content":"正确、完整且简洁的回答"}],"rejected":[{"role":"assistant","content":"包含明确概念错误的回答"}]}
```

数据工作至少包含：

- 明确许可和来源；去 PII、去重复、事实审核。
- 按来源/主题/实体切分，避免模板变体泄漏。
- 统计 token 长度、截断比例、主题和难度。
- 建立独立测试集：规则可判定题 + 人工 rubric 题。
- 偏好数据控制长度偏置；chosen 不能总比 rejected 长。

### 2.4 QLoRA 配置基线

复制当前官方 Qwen3 LoRA YAML 再修改，不从网上复制旧版本。下面仅说明应出现的语义字段；字段名以固定 commit 为准：

```yaml
model_name_or_path: Qwen/Qwen3-4B
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
quantization_bit: 4
quantization_method: bitsandbytes
dataset: my_domain_sft
template: qwen3
cutoff_len: 2048
packing: true
learning_rate: 0.0001
num_train_epochs: 2.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
logging_steps: 5
save_steps: 200
output_dir: saves/qwen3-4b/domain-qlora
```

运行前检查：模型卡要求的 chat template、显存、BF16 支持；用 20 条数据跑 10 step；打印可训练参数；手动 decode 一个 tokenized 样本确认 label mask。

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/qwen_domain_qlora.yaml
# 多卡启动方式以当前 examples 为准；常见入口：
FORCE_TORCHRUN=1 llamafactory-cli train configs/qwen_domain_qlora.yaml
```

### 2.5 DPO 和公平评测

先选 SFT 最佳 checkpoint 作为 policy，再创建 DPO 配置。保持测试集、模板和解码一致，比较：

1. Base。
2. SFT/QLoRA。
3. SFT + DPO，不同 beta 至少 2 个点。
4. 可选：同预算全精度 LoRA 或不同 rank。

至少记录：任务正确率、格式合法率、拒答率、长度、人工偏好胜率、通用能力回归、峰值显存、训练时长。保存逐样本输出，做错误分桶。

不要让 LLM judge 知道模型名字；A/B 顺序随机并交换复评。对 50–100 条样本做人工校准。

### 2.6 合并与服务

LoRA adapter 小且可切换；合并到非量化基座后部署更简单，但失去动态 adapter，且需足够 CPU/GPU 内存。使用当前仓库 `examples/merge_lora` 配置和 `llamafactory-cli export`，不要直接把 4-bit 量化 Tensor 当普通权重相加。

官方当前快速路径示例为：

```bash
llamafactory-cli chat configs/qwen_domain_infer.yaml
llamafactory-cli export configs/qwen_domain_merge.yaml
# 服务入口在不同 commit 可能变化，读取仓库 Deploy 章节；也可用 vLLM：
vllm serve ./exports/qwen-domain-merged --host 0.0.0.0 --port 8000
```

服务验收：健康检查、并发 1/4/8、固定输入/输出长度下的 TTFT/TPOT/P95、错误请求、超长输入和显存。不要把端口直接暴露公网；加鉴权和安全组。

### 2.7 DeepSpeed 对照

在同一模型、数据和 global batch 下比较 ZeRO-1/2/3；记录每卡峰值显存、总 tokens/s、step time、通信占比和 checkpoint 大小。ZeRO-3 若更慢不是“配置失败”，可能是参数 All-Gather 成本超过收益。

### 2.8 完成等级与排错

- L1：跑通官方 LoRA 示例。
- L2：自有数据 QLoRA、评测、adapter 推理。
- L3：SFT/DPO 或 rank/beta 公平对照与错误分析。
- L4：合并/API/压测/DeepSpeed 报告/数据卡齐全。

常见故障：模板错误导致输出异常；`loss=0/NaN` 检查 labels；OOM 先减长度/micro-batch；训练很好测试差检查泄漏；merge 后效果变检查基座版本、template 和 dtype。

### 2.9 高频扩展：RM/GRPO、蒸馏与量化部署

这些是选修实验；先完成可信 SFT/DPO 基线。RL 实验优先选择答案可由规则验证的小任务（数学格式、SQL/代码单测、结构化抽取），不要用主观 judge reward 直接得出“推理能力提升”。

**A. Reward Model 与 GRPO**

- 从同一 prompt 的回答构造 chosen/rejected，检查长度、模板和位置偏置；训练 RM 时保存 pair accuracy、swap consistency 与长度分桶。
- GRPO 对每个 prompt 采样 `G` 个回答，记录 reward mean/std、全相同 reward 组比例、KL、clip fraction、response length 和 rollout 吞吐。
- 对比 `SFT → DPO` 与 `SFT → GRPO` 时固定基座、测试集、总训练/rollout 预算和解码；GRPO 配置应注明所用 TRL/LLaMA-Factory commit，因为实现细节会变化。
- GSPO/DAPO 只做论文复现选修：先写出与 GRPO 的目标/采样差异，若没跑就不能写入简历成果。

**B. 蒸馏**

用更强教师为训练集生成多个候选，经规则/人工过滤后做 sequence-level distillation；可选保存教师 logits 做温度 KL。对照“原始 SFT 数据”“教师未过滤数据”“过滤后蒸馏数据”，分析教师错误与学生错误的重合。

**C. 量化与服务**

合并后的 BF16/FP16 模型分别生成 GPTQ/AWQ（若模型与硬件支持）或使用受支持的 weight-only 量化；GGUF 仅在 llama.cpp 路径作为部署格式实验。固定 200–500 条质量集和压测长度分布，记录：磁盘/HBM、加载时间、任务分数、TTFT、TPOT、P95、吞吐。禁止把不同 engine 的默认 chat template/采样参数直接比较。

**D. 模型架构说明**

生成 `reports/model_card.md`，从实际 `config.json` 列出层数、hidden、MHA/GQA、KV heads、RoPE、FFN/MoE、词表和 context；明确这是哪个 Qwen checkpoint，而非“Qwen 系列都如此”。

**真实面试追问**：RM pair accuracy 高为何仍会 reward hacking？GRPO 组内 reward 全相同会怎样？为什么选择 DPO 而非 GRPO？量化为何没提速？adapter merge 前后如何验证等价？  
**必须保存的证据**：数据卡、训练/rollout 配置与 commit、reward/KL/length 曲线、逐样本输出、人工校准集、量化模型 hash、质量—显存—延迟表、服务压测命令。

### 2.10 面试与简历

```text
- 基于 Qwen [准确版本] 与 LLaMA-Factory，在自建 [领域/数量] 指令数据上完成
  QLoRA SFT 与 DPO；通过 [数据去重/划分方法] 控制泄漏，在 [测试样本数] 上将
  [指标] 从 [A] 提升至 [B]，并分析 [主要错误类型]；使用 [GPU×数量] 对比
  DeepSpeed ZeRO [阶段] 的显存和吞吐，最终以 vLLM 部署，P95 为 [实测值]。
```

---

## 项目 3：带分层评测的企业知识库 RAG

### 3.1 项目目标

做一个可追溯、可增量更新、能分层定位错误的知识助手。建议选 100–1000 篇公开技术文档或课程资料，避免公司机密。核心亮点不是聊天界面，而是 gold evidence 测试集、混合检索、rerank、引用和错误分析。

### 3.2 架构

```text
                   ┌─ BM25 ─────┐
文档→解析→结构化块─┤             ├→RRF→Reranker→Context→LLM→引用校验
                   └─ Vector DB ┘
             元数据/版本/权限             │
问题→规范化/改写──────────────────────────┘
```

实现时让领域逻辑保持纯 Python：`Document`、`Chunk`、`Retriever`、`Reranker` 定义自己的接口，LangChain/LlamaIndex 只作为适配器。这样面试时能解释底层，也降低框架 API 变化影响。

### 3.3 环境与数据 schema

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U fastapi uvicorn pydantic numpy pandas
pip install sentence-transformers rank-bm25 qdrant-client
pip install langchain langgraph langchain-openai
```

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    version: str
    title: str
    text: str
    token_count: int
    source_url: str
    acl: list[str]
```

`chunk_id` 应稳定且可追到原文；删除/更新文档时按 `doc_id+version` 处理旧索引。公开作品仍应记录许可和抓取日期。

### 3.4 建测试集再调系统

手工构建至少 100 条：

```json
{"id":"q001","question":"ZeRO-2 分片哪些状态？","answer":"优化器状态和梯度","gold_chunk_ids":["deepspeed_zero_002"],"category":"exact","difficulty":"easy"}
```

建议分桶：精确术语、跨段落、多文档比较、时效问题、不可回答、冲突文档、Prompt Injection。每题保存 gold evidence，不只保存答案。

### 3.5 分阶段实现

**Baseline A：BM25**。验证解析和 gold chunk；记录 Recall@5/10。  
**Baseline B：Dense**。选择支持中文/多语的 embedding 模型（如 BGE-M3，运行前检查模型卡许可和 pooling），批量归一化 embedding。  
**Hybrid**：用 Reciprocal Rank Fusion：

$$RRF(d)=\sum_i\frac{1}{k+rank_i(d)}$$

**Rerank**：只对前 20–50 个候选用 cross-encoder 排序，再取 3–8 个进入 context。  
**Generation**：Prompt 明确“仅基于证据、逐条引用、证据不足则拒答”。输出结构化 `answer + citations`，服务端验证 citation id 是否确实在 context 中。

```python
def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    objects = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            objects[item.chunk_id] = item
    return sorted(objects.values(), key=lambda x: scores[x.chunk_id], reverse=True)
```

### 3.6 Context Engineering

- 去重近似块，保留标题、日期、来源和 chunk id。
- 按证据相关性与时效排序；冲突时明确展示两个来源。
- 控制 token budget，为问题、指令、证据、输出分别预算。
- 外部文档视为不可信数据；其中“忽略系统提示”等文本不得成为指令。
- 历史对话先改写成独立 query，但保留原问用于生成。

### 3.7 评测与实验矩阵

| 层 | 指标 | 说明 |
|---|---|---|
| 解析 | parse success、空块/乱码率 | 保证输入正确 |
| 检索 | Recall@K、MRR、nDCG | gold evidence 是否进入候选 |
| 重排 | Recall@K/排名提升 | 不让 gold 被精排丢掉 |
| 生成 | correctness、faithfulness | 正确与证据支持分开 |
| 引用 | citation precision/recall | 引用是否真的支撑陈述 |
| 系统 | P50/P95、tokens、cost | 在固定负载口径下报告 |

实验：chunk 256/512/1024；overlap 0/15%；BM25/dense/hybrid；reranker on/off；query rewrite on/off。一次只改变一类变量。

`lm-evaluation-harness` 适合通用 LLM benchmark 和自定义任务；RAG 检索指标建议自己实现或使用专用评测框架，并保留逐样本结果。参考 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的 YAML task 与结果记录方式。

### 3.8 API 与故障注入

```text
POST /v1/ask
request:  {"question": "...", "session_id": "...", "user_groups": ["public"]}
response: {"answer": "...", "citations": [{"chunk_id":"...","url":"..."}],
           "trace_id":"...", "latency_ms": 1234}
```

测试：空 query、超长 query、文档不存在、embedding 服务超时、向量库不可用、LLM 超时、恶意文档注入、越权 group、文档删除后缓存未失效。降级策略必须明确，不能在没有证据时悄悄让模型自由回答。

### 3.9 高频扩展：多格式解析、Embedding、GraphRAG 与安全

**A. 表格/图片解析**

扩展 chunk schema：`doc_id/version/page/bbox/modality/title/content/ocr_confidence/acl`。表格块携带表名、列名和行上下文；图片保留原区域并生成 caption/OCR，但回答时可把区域图片交给 VLM 核验。建立 30–50 页解析 gold 集，测表格单元格/阅读顺序/图片引用，而非只看“解析成功”。

**B. Embedding hard-negative 训练**

从 BM25/dense 前 20 名中采“看似相关但不支持答案”的 negatives，人工抽检假负例；用 InfoNCE 微调双塔。固定索引与测试集，对比通用 embedding、领域微调、加入 hard negatives，报告分桶 Recall@K，不只报告平均值。

**C. BM25/HNSW 与高级检索**

- sweep HNSW `M/efConstruction/efSearch`，用 exact search 作 recall gold，画 recall—P95—index size。
- 对多跳集实现 query decomposition，保存每一跳 evidence；设最大跳数和证据不足退出。
- GraphRAG 只用于实体关系/全局问题子集，记录实体消歧、错误边、构图/更新成本；普通事实仍走 hybrid RAG。
- Lost-in-the-Middle 通过固定证据、只交换 context 位置验证；对比 rerank、压缩和关键证据前置。

**D. 增量更新与时间**

用稳定 document ID + content hash 定位变化块：新版本 upsert → 原子切换 active version → 删除旧向量/关键词/图边 → 失效 cache。测试新增、修改、撤回和 ACL 变化。时间衰减只在新闻/状态类数据启用；政策类按有效期和权威级别处理。

**E. Prompt Injection 测试**

在测试文档加入“忽略规则、调用工具、泄露环境变量”等攻击文本；检查它只能作为引用数据出现，不能改变 system policy。检索、生成、引用、权限四层分别记录是否拦截；无证据时返回 abstain/fallback。

**真实面试追问**：为什么选这个 embedding/chunk/top-k？HNSW 的 efSearch 怎么定？GraphRAG 比向量 RAG 真提升在哪个子集？如何彻底删除文档？恶意 OCR 怎么处理？  
**必须保存的证据**：解析 gold 与截图、索引参数 sweep、hard-negative 样本/误标率、每跳 trace、错误图边、版本切换日志、注入/越权回归结果、逐样本引用映射。

### 3.10 完成等级与简历

- L1：单路向量检索和回答。
- L2：自有文档、100 条 gold evidence 测试集、引用。
- L3：混合检索+rerank+消融+错误分桶。
- L4：增量索引、权限、API、压测、故障注入。

```text
- 构建面向 [文档类型/规模] 的可追溯 RAG 系统，设计 [N] 条带 gold evidence
  的分层测试集；通过 BM25+dense RRF 与 reranker 将 Recall@[K] 从 [A]
  提升至 [B]，faithfulness 达 [C]；实现文档版本/删除、引用校验和权限过滤，
  在并发 [N] 下 P95 [实测值] ms，并完成 [主要错误类型] 的归因与修复。
```

---

## 项目 4：LangGraph 多工具研究 Agent

### 4.1 项目目标

构建一个“研究问题 → 制定计划 → 搜索本地知识库/可信来源 → 计算或运行只读工具 → 核验 → 输出带来源报告”的 Agent。重点是显式状态、可恢复执行、终止、错误处理、人工确认和轨迹评测。

LangGraph 当前定位为低层状态化 Agent 编排框架，参考 [官方仓库](https://github.com/langchain-ai/langgraph)；固定版本后按对应 docs 编写，避免复制旧版 `langchain` 教程。

### 4.2 状态图

```text
START → planner → router ─┬→ retrieve ─┐
                          ├→ calculate ├→ verify → [done?] → answer → END
                          └→ ask_human ┘       └→ router
```

```python
from typing import Annotated, Literal, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict):
    question: str
    plan: list[str]
    step: int
    evidence: Annotated[list[dict], add]
    tool_errors: Annotated[list[dict], add]
    final_answer: str
    status: Literal["running", "needs_human", "done", "failed"]

graph = StateGraph(ResearchState)
graph.add_node("planner", planner)
graph.add_node("retrieve", retrieve)
graph.add_node("verify", verify)
graph.add_node("answer", answer)
graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", route_next, {"retrieve": "retrieve", "human": END})
graph.add_edge("retrieve", "verify")
graph.add_conditional_edges("verify", after_verify, {"continue": "planner", "done": "answer"})
graph.add_edge("answer", END)
app = graph.compile(checkpointer=checkpointer)
```

这只是骨架；运行时使用所锁版本的持久化 checkpointer。节点函数应小而可测试，模型只输出结构化决策，工具执行由代码完成。

### 4.3 工具设计

至少四个工具：

1. `retrieve_documents(query, filters)`：调用项目 3。
2. `calculator(expression)`：用安全表达式解析器，不直接 `eval`。
3. `get_document(chunk_id)`：获取完整证据。
4. `save_draft(report)`：只写 sandbox；最终导出前人工确认。

每个工具定义：schema、权限、超时、最大返回长度、错误码、是否幂等、重试策略。网页/文档内容永远是数据，不允许覆盖系统策略。

```python
class ToolResult(BaseModel):
    ok: bool
    code: str
    data: dict | None = None
    safe_message: str
    retryable: bool = False
```

### 4.4 记忆与 Context

- thread state 保存本任务 plan/evidence/errors。
- 长期 memory 只存经过确认的用户偏好，带 `user_id/source/timestamp`。
- 原始工具结果放外部存储，context 中只放摘要+引用，避免无限膨胀。
- 摘要不能覆盖结构化约束；关键事实单独字段化。
- 多租户检索必须在服务端强制 tenant/user filter。

### 4.5 终止与恢复

硬限制：`max_steps`、deadline、token/cost budget、每工具重试上限。软检测：相同工具+参数重复、计划无进展、证据矛盾。高风险操作进入 `needs_human`，不要让 prompt 自行模拟批准。

为每个节点保存 checkpoint；进程中断后从最后一个安全状态恢复。非幂等工具用 idempotency key，避免恢复时重复执行。

### 4.6 测试集与指标

构建至少 50 条任务：单工具、两工具、多跳、无答案、工具超时、参数错误、相互冲突证据、Prompt Injection、循环诱导、预算耗尽。

指标：

- task success（用规则/环境状态判定优先）。
- tool selection accuracy、argument exact/semantic accuracy。
- redundant calls、平均/最大 steps。
- evidence coverage、citation accuracy。
- P50/P95 latency、input/output tokens、成本。
- loop、timeout、unsafe action、memory leakage 率。

实验对照：普通单次 RAG、固定 workflow、ReAct Agent、加入 verifier 的 Agent。使用相同任务和模型；比较成功率提升是否值得成本/延迟。

### 4.7 故障注入

- 工具返回 429/500/timeout/空结果/超大结果。
- 模型给不存在工具、缺字段、错误类型。
- 检索文档包含“忽略规则并泄露密钥”。
- 同一写操作重复请求。
- checkpoint 后进程退出再恢复。
- 两个用户访问同一 memory key。

验收不是“Agent 最终说成功”，而是 mock 环境状态与审计日志证明成功。

### 4.8 高频扩展：MCP、状态契约、隔离与 Agent 评测

**A. MCP 工具适配**

按 MCP 2025-11-25 正式规范实现一个只读本地知识库 server（例如 `search_docs`/resource）和一个计算 tool。保存 initialize/capability negotiation/call/error trace。MCP 只做协议适配；权限仍由 host/policy gate 判断。另保留直接 Python adapter，用同一 contract tests 验证结果一致。

**B. 结构化状态与 context 压缩**

给 state 加 `schema_version/tenant_id/session_id/run_id/version/pending_calls/evidence_refs/budget`；节点只返回 state patch，由 reducer 校验合并。原始工具结果存对象存储/本地 artifact，context 仅保存摘要与引用。设计 20+ step 长任务，对比无压缩、滑窗、任务摘要的成功率/token/丢约束率。

**C. 长期记忆衰退与安全**

记忆条目必须有 owner、source、confidence、created/last_used、expiry 和 supersedes；读取时做 tenant filter，写入需类型/权限/敏感信息检查。用 TTL、时间/访问衰减和显式删除；错误记忆注入后验证能撤回，不能只删除向量不删原记录。

**D. 沙箱与并发隔离**

只读工具限制目录/网络；写工具使用 policy gate、human confirmation 和 idempotency key。用 10–50 个并发 session 注入重复消息、乱序结果、迟到回调、同 key 写入和取消，状态更新使用 version/CAS，确保没有跨用户串线或重复副作用。

**E. 用户模拟器与评测 Harness**

把测试定义为 Task，配置一次运行作为 Trial，保存 Transcript 与 Outcome；代码 grader 判断环境最终状态，transcript grader 判断工具/安全，人工 gold 校准模型 grader。每 task 至少多次 trial，报告 pass@1、pass@k、pass^k、均值成本和 CI；固定容器、mock 工具、随机种子与数据 snapshot，并在 trial 后还原环境。

**真实面试追问**：MCP 与 function calling/A2A 的边界？摘要丢了否定约束怎么办？并发迟到结果如何处理？结果正确但过程越权怎么算？pass@k 高而 pass^k 低说明什么？  
**必须保存的证据**：协议版本与 trace、state JSON Schema、replay/resume 测试、压缩前后对照、memory 删除审计、sandbox policy、并发故障日志、每 trial transcript/outcome、grader 校准报告。

### 4.9 完成等级与简历

- L1：LangGraph 两节点流程可运行。
- L2：四个工具、持久化 state、50 条测试集。
- L3：RAG/workflow/Agent 对照、失败分类和 verifier。
- L4：API/UI、可观测 trace、故障注入、安全和恢复。

```text
- 使用 LangGraph 构建面向 [研究任务] 的有状态多工具 Agent，设计 planner、
  retriever、verifier 与 human-in-the-loop 节点，实现 checkpoint 恢复、幂等重试和
  循环检测；在 [N] 条任务上成功率 [A]、工具参数准确率 [B]，相较普通 RAG
  提升 [C]，同时将平均步骤/成本控制在 [真实值]。
```

---

## 项目 5：InternVL / Qwen-VL 多模态问答微调

### 5.1 项目目标

选择一个具体、易评测的视觉领域：课程讲义图表问答、公开票据字段抽取、设备面板读数或公开科研图表理解。完成 base 推理、图文数据清洗、LoRA/projector 微调、分桶评测和服务 Demo。

优先使用 [InternVL 官方仓库](https://github.com/OpenGVLab/InternVL) 的同一版本完成 Quick Start → Finetune → Evaluate → Deploy；若选择 Qwen-VL，所有模板和预处理改用对应官方仓库，不能混用。

### 5.2 固定仓库与环境

```bash
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL
git rev-parse HEAD | tee USED_COMMIT.txt
# 按该 commit README 创建官方建议环境；推荐容器/独立 venv。
nvidia-smi | tee reports/gpu.txt
python -m pip freeze > requirements-lock.txt
```

InternVL 版本和训练引擎变化较快，不在本手册硬编码可能过期的完整 launch 参数。执行步骤：

1. 在 README 选择仍维护且资源可承受的 2B/4B/8B 级 checkpoint。
2. 跑官方 single-image chat，保存输入、原始输出、显存和耗时。
3. 跑官方提供的最小 finetune recipe，固定 commit 与 config。
4. 复制 recipe 到自己的 `configs/`，只逐项修改数据、模型和资源参数。

### 5.3 数据格式与质量

每条样本保存图片路径、对话、任务类别、来源和可验证标注。实际 JSON schema 以固定 commit 的数据说明为准。概念示例：

```json
{"id":"chart_001","image":"images/chart_001.png","conversations":[{"from":"human","value":"<image> 图中 2025 年的值是多少？"},{"from":"gpt","value":"42。"}],"meta":{"task":"chart_qa","source":"self_created"}}
```

清洗清单：

- 图片可解码、方向正确、无重复/近重复、许可清楚。
- 答案确实可从图中得到；OCR 字符和单位人工抽检。
- `<image>` 占位符数量和顺序与图片一致。
- 按原始文档/模板划分 train/test，避免相似页面泄漏。
- 统计分辨率、长宽比、视觉 token、问题/答案长度。
- 加入“图中不存在”的负样本，降低强猜。

### 5.4 先理解 tensor shape

运行一个 batch 并记录：原图尺寸 → 预处理尺寸/tiles → vision encoder 输出 `[B,Nv,Dv]` → projector 输出 `[B,Nv,Dllm]` → 与文本 tokens 合并后的长度。回答以下问题后再训练：

- 动态分辨率如何决定 tiles？
- 每张图产生多少视觉 token？
- 多图时 token budget 如何增长？
- 哪些模块被冻结？各模块可训练参数量和学习率？

### 5.5 三组训练对照

1. **Projector-only**：冻结 vision tower 与 LLM，低成本验证模态对齐。
2. **Projector + LLM LoRA**：提高任务适配，控制语言能力回退。
3. **部分解冻 vision tower**：仅当领域视觉差异大且有足够数据/显存。

保持数据、步数、有效 batch、分辨率和评测相同。记录 trainable params、峰值显存、images/s、训练时长和各任务分桶指标。

### 5.6 评测

| 类别 | 指标/方法 |
|---|---|
| 字段抽取 | Exact Match、字段级 F1、JSON 合法率 |
| OCR/数值 | 规范化 EM、单位正确率、数值容差 |
| 图表问答 | 正确率，按检索/计算/比较分桶 |
| 描述 | 人工 rubric + 校准后的 judge |
| 幻觉 | 不存在对象误报率、遮图/换图性能下降 |
| 系统 | 每图延迟、显存、最大分辨率/图片数 |

反事实测试非常关键：遮住答案区域、替换图片、只输入问题文本。若模型仍高分，说明可能利用数据偏差而非视觉。

### 5.7 多卡和显存排错

VLM OOM 常来自高分辨率带来的视觉 token/激活，不只是 LLM 参数。依次尝试：降低 max tiles/分辨率和 micro-batch、gradient checkpointing、BF16、冻结 vision tower、LoRA、ZeRO。变更分辨率后必须重测质量。

多卡出现负载不均时统计每 rank 的视觉 token 数；按样本数均分不等于按计算量均分。必要时按图像成本 bucketing。

### 5.8 服务 Demo

提供上传图片+问题的 Web/API，返回答案、耗时和模型版本。限制文件类型/大小、像素总量和图片数；图片在隔离目录处理并定期删除。不要允许用户上传任意路径或让模型执行图片中的指令。

### 5.9 高频扩展：CLIP、Grounding、高分辨率与多模态 RAG

**A. CLIP/InfoNCE 小实验**

选公开图文对，冻结两个 encoder 只训练 projection/temperature，或直接用预训练 CLIP 做检索。实现双向 InfoNCE，与手写 similarity matrix 对齐；记录 image→text/text→image Recall@K、batch/temperature、假负例。该实验用于理解对齐，不宣称训练了通用 CLIP。

**B. Grounding 与视觉反事实**

若数据有 bbox/point，统一坐标归一化和 resize/pad 反变换，测 IoU/point accuracy；保存预测区域可视化。建立四种反事实：遮目标、换成近似图、OCR 文本与图像冲突、打乱多图顺序。将“答案正确但定位错”单独计数。

**C. 高分辨率策略**

对比固定 resize、thumbnail+tiles、官方 dynamic resolution；固定模型与问题，记录每图 tiles/视觉 token、峰值显存、images/s、P95 和小字/图表分桶分数。多卡 sampler 按视觉 token cost bucketing，记录各 rank token 数，不只看样本数。

**D. 多模态 RAG（选修）**

同时索引 OCR/页面文本、表格结构和图像/区域 embedding；query 路由后混合召回与 rerank，返回原页面 crop + 文本给 VLM，并输出 `doc_id/page/bbox` 引用。对比 text-only、image-only、multimodal；评跨模态 Recall、引用区域正确率、端到端答案与注入安全。

**真实面试追问**：CLIP 为何不能直接生成？模型是否真的看图？分辨率翻倍 token 为什么约四倍？冻结 vision tower 的依据？caption 检索为何不足？  
**必须保存的证据**：图文数据卡、InfoNCE 单测/曲线、grounding 可视化、反事实逐样本结果、resolution—token—显存—质量表、多模态索引 schema 与区域引用截图。

### 5.10 完成等级与简历

- L1：官方单图推理和评测样例。
- L2：自建数据、projector/LoRA 微调、独立测试。
- L3：冻结策略对照、反事实/幻觉评测、错误图库。
- L4：多卡训练报告、API/UI、安全限制和复现文档。

```text
- 基于 InternVL [准确版本] 构建 [视觉领域] 多模态问答系统，清洗并标注
  [N] 条图文数据，对比 projector-only、LLM LoRA 与 [解冻策略]；在独立测试集上
  [指标] 从 [A] 提升至 [B]，通过遮图/换图反事实测试将视觉幻觉率量化为 [C]；
  使用 [GPU×数量] 训练并部署图片问答 API，单图 P95 [实测值]。
```

---

## 6. 附加实验：DeepSpeed ZeRO 与 Megatron 两卡 TP

### 6.1 ZeRO 公平对照

用项目 1 或 2 的同一模型：

```text
固定：模型、序列长度、global batch、训练 step、precision、数据顺序
改变：DDP / ZeRO-1 / ZeRO-2 / ZeRO-3
记录：allocated/reserved/peak GB、tokens/s、step P50/P95、通信时间、checkpoint 大小
```

参考 [DeepSpeed ZeRO 官方教程](https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/zero.md)。结果表必须解释 stage 越高为何可能更慢，以及何种模型规模下收益才出现。

### 6.2 Megatron 两卡 TP 最小实验

使用 [Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM) 当前 `Megatron Core Quick Start`，在官方推荐 NGC 容器中运行 2-GPU mock GPT：

1. 固定仓库 commit 和容器 tag。
2. `tensor_model_parallel_size=2`、`pipeline_model_parallel_size=1`。
3. 跑 5–20 iteration，保存 profiler/NCCL 日志。
4. 与单卡可容纳的小模型对比，解释 All-Reduce/Reduce-Scatter 出现位置。
5. 修改隐藏维或序列长度，观察 compute/communication ratio。

目标不是声称“训练了百亿模型”，而是能根据 timeline 解释 TP 如何切列并行/行并行线性层、为什么偏好节点内高速互联。

---

## 7. 8–12 周执行排期

| 周 | 项目工作 | 周末验收 |
|---|---|---|
| 1 | 项目 1 数据、模型、单测 | 单 batch 过拟合 + Attention 对齐 |
| 2 | 项目 1 完训、AMP/DDP 消融 | 报告和可复现命令 |
| 3 | 项目 2 数据/基线/QLoRA | 10-step smoke + 完整训练 |
| 4 | 项目 2 DPO/评测/部署 | 错误分析 + API 压测 |
| 5 | 项目 3 解析/BM25/dense | 100 条 gold evidence 集 |
| 6 | 项目 3 hybrid/rerank/API | 分层指标 + 故障测试 |
| 7 | 项目 4 图/工具/状态 | 50 条 Agent 任务集 |
| 8 | 项目 4 verifier/恢复/评测 | RAG/workflow/Agent 对照 |
| 9 | 项目 5 推理/数据/基线 | shape 记录 + base 分桶 |
| 10 | 项目 5 微调/反事实评测 | 冻结策略实验 + Demo |
| 11 | ZeRO/Megatron 附加实验 | 多卡性能报告 |
| 12 | 清理仓库、录 Demo、模拟面试 | 所有数字可追溯、README 可复现 |

资源不足时优先完成项目 1–4 的 L3，再把项目 5 做到 L2；不要五个项目都只停在 L1。

---

## 8. 招聘要求覆盖矩阵

| 招聘技术要求 | 本地资料高频主题 | 技术指南 | 面试题 | 项目证据 |
|---|---|---|---|---|
| Python、PyTorch、模型开发 | 训练循环、手写题、排障 | §1–2 | Q1–Q13 | 项目 1 |
| Transformer、BERT/LLaMA/Qwen | Attention、RoPE、模型对比 | §2 | Q6–Q13 | 项目 1/2 |
| 数据处理、实验分析 | 数据质量、评测、项目深挖 | §3、§7 | Q14–Q18、Q41 | 全部项目 |
| SFT、LoRA、QLoRA | 参数量、显存、数据格式 | §4 | Q19–Q21 | 项目 2 |
| DPO、RLHF、Agentic RL、OPD | 目标函数、reward、偏好数据 | §4、§11 | Q21–Q26 | 项目 2；进阶阅读 |
| AMP、裁剪、训练优化 | OOM、NaN、梯度异常 | §5 | Q27–Q32 | 项目 1/2 |
| Linux、Git、Hugging Face | 环境与复现、框架调用 | §1 | Q4 | 全部项目环境证据 |
| DDP、DeepSpeed、Megatron | ZeRO、显存、通信与吞吐 | §6 | Q33–Q38 | 项目 1/2、附加实验 |
| Prompt、模型评测 | Context、judge、指标口径 | §7 | Q39–Q41 | 项目 2–5 |
| RAG、Context Engineering | 召回、rerank、幻觉与排障 | §8 | Q42–Q46 | 项目 3 |
| Agent、记忆、Tool Use、ReAct | 状态、循环、工具失败 | §9 | Q47–Q53 | 项目 4 |
| LangChain/LangGraph/LlamaIndex/AutoGen | 框架定位与选型 | §9 | Q52 | 项目 3/4 |
| 多模态融合与对齐 | projector、冻结、幻觉 | §10 | Q54–Q58 | 项目 5 |
| 世界模型、论文复现 | 论文解读与开放题 | §11 | Q59–Q60 | 项目 1/5 复现报告 |
| 数学、优化与指标 | Softmax/CE、AdamW、AUC/F1 | §1.5 | Q67–Q74 | 项目 1 数学单测；全项目评测 |
| MQA/GQA/MLA、FlashAttention、MoE | KV Cache、Flash、MoE 路由 | §2.6–2.7 | Q75–Q84 | 项目 1 扩展；项目 2 模型卡 |
| PEFT、蒸馏、RM、GRPO/GSPO/DAPO | GRPO 对比、RM、蒸馏 | §4.6–4.7 | Q85–Q94 | 项目 2 扩展实验 |
| Prefill/Decode、PagedAttention、量化、FSDP | TTFT/TPOT、量化、显存粗估 | §5.5–6.5 | Q95–Q102 | 项目 1/2 服务与多卡报告 |
| Embedding、高级 RAG、GraphRAG | BM25/HNSW、多跳、更新 | §8.5 | Q103–Q112 | 项目 3 扩展实验 |
| MCP/A2A、沙箱、并发与 Agent 评测 | 协议、状态隔离、pass@k/pass^k | §9.5–9.6 | Q113–Q124 | 项目 4 扩展实验 |
| CLIP、Grounding、视频、多模态 RAG | InfoNCE、高分辨率、视觉反事实 | §10.4 | Q125–Q130 | 项目 5 扩展实验 |

---

## 9. 项目完成后如何写进简历

这一章的目标不是把项目“包装得像大厂系统”，而是把真实完成的工作压缩成招聘者能快速理解、面试官可以继续深挖的项目经历。推荐从 5 个项目中选择与岗位最匹配、完成度最高的 2–3 个写入一页简历；其余项目放在 GitHub 主页或“其他项目”中，不要让五个低完成度项目挤占版面。

### 9.1 先根据岗位选择项目组合

| 目标岗位侧重 | 推荐主项目 | 可作为补充 | 简历需要突出 |
|---|---|---|---|
| LLM 训练/后训练算法 | 项目 2 + 项目 1 | ZeRO/Megatron 附加实验 | 数据、loss、LoRA/DPO/GRPO、消融、训练稳定性 |
| 大模型算法综合岗 | 项目 2 + 项目 3 + 项目 1 | 项目 4 | 训练、评测、RAG、部署的完整能力链 |
| RAG/LLM 应用算法 | 项目 3 + 项目 4 | 项目 2 | 检索指标、错误归因、Agent 状态与可靠性 |
| Agent 算法/平台 | 项目 4 + 项目 3 | 项目 2 | 工具协议、状态、记忆、安全、评测 Harness |
| 推理部署/性能优化 | 项目 1 + 项目 2 | ZeRO/Megatron、项目 4 | 显存、吞吐、TTFT/TPOT、量化、并行与压测 |
| 多模态算法 | 项目 5 + 项目 2 | 项目 3 的多模态 RAG | 图文数据、对齐、冻结策略、Grounding、反事实评测 |

选择原则：优先写达到 L3/L4、能拿出代码和实验记录的项目。L1 只是运行官方 Demo，不适合写成核心项目；L2 可以写“完成基线和自定义数据实验”；L3 才适合强调方案比较、消融和错误分析；L4 可以进一步强调 API、压测、可靠性和复现工程。

### 9.2 一段项目经历的标准结构

推荐每个项目占 4–6 行，使用下面的结构：

```text
项目名称｜个人项目/课程项目｜起止时间                       GitHub / Demo
一句话背景：为 [用户/任务] 解决 [具体问题]，数据或使用场景为 [真实范围]。
技术栈：PyTorch、Transformers、PEFT、vLLM……
- 实现：你亲自完成的核心模块、数据流程或算法，不罗列无关框架。
- 实验：基线、对照、消融、测试集和关键结果，说明指标口径。
- 工程：部署、性能、故障处理、复现或安全；没有做过则不写。
```

每条 bullet 尽量符合“动作 + 对象 + 方法 + 结果/验证”结构。例如：

```text
实现 BM25 与 dense 双路召回并通过 RRF 融合，加入 cross-encoder reranker；
在 [N] 条带 gold evidence 的测试集上，将 Recall@[K] 从 [A] 提升到 [B]，
并通过逐样本错误分析确认主要收益来自 [真实问题类型]。
```

不要只写“使用 LangChain 搭建 RAG 系统”。这只说明调用了框架，没有说明你解决了什么问题、为什么这样设计、效果是否可信。

### 9.3 哪些数字可以写，应该如何写

所有数字必须能回到日志、配置或评测文件。一个完整指标至少包含“名称、数值、测试规模、实验条件”中的前三项；性能数字还必须给出硬件、并发和长度分布。

| 指标类型 | 合格写法 | 不合格写法 |
|---|---|---|
| 训练效果 | 在独立 `[N]` 条测试集上，字段 F1 从 `[A]` 提升至 `[B]` | 效果显著提升 |
| 检索效果 | Recall@5 从 `[A]` 提升至 `[B]`，测试集含 `[N]` 条 gold evidence query | 检索准确率 95%（未定义准确率） |
| Agent | `[N]` 个任务、每个 `[K]` 次 trial，pass@1 为 `[A]`、pass^`[K]` 为 `[B]` | Agent 成功率很高 |
| 显存 | 在 `[GPU 型号]`、sequence `[T]` 下峰值显存由 `[A]` GB 降至 `[B]` GB | 显存降低 4 倍（只按位宽推算） |
| 推理性能 | 并发 `[C]`、输入/输出 P50 长度 `[X]/[Y]` 下，P95 TTFT `[A]` ms、TPOT `[B]` ms | 推理速度提升 200%（无负载口径） |
| 多卡扩展 | 1/2/4 卡总吞吐为 `[A/B/C]` tokens/s，4 卡扩展效率 `[E]%` | 多卡线性加速 |

如果没有测某项，保留在项目报告的“下一步”中，不要在简历上用理论估算冒充实测。百分比要说明基线；“提升 20%”应明确是相对提升还是增加 20 个百分点。

### 9.4 项目 1：Mini Transformer 简历模板

**建议项目名称**：`从零实现 Decoder-only Mini Transformer 与训练性能分析`

**适合写入的前提**：至少达到 L2；若要写 DDP、FlashAttention、GQA 或 MoE，必须实际完成相应实验。

**技术栈候选**：Python、PyTorch、RoPE、RMSNorm、SwiGLU、SDPA/FlashAttention、AMP、DDP、DeepSpeed。只保留实际使用的部分。

```text
从零实现 Decoder-only Mini Transformer 与训练性能分析｜个人项目｜[时间]
技术栈：PyTorch、[RoPE/RMSNorm/SwiGLU]、[AMP/DDP/DeepSpeed]
- 独立实现 Tokenizer 接入、Causal Self-Attention、Transformer Block、训练与
  自回归生成，模型规模为 [参数量]，在 [数据集/Token 数] 上完成训练。
- 编写 causal mask、SDPA 数值对齐、单 batch 过拟合和 checkpoint resume 等
  [N] 项测试；完成 [Pre/Post-Norm、GELU/SwiGLU、MHA/GQA] 消融，观察到 [真实结论]。
- 在 [GPU×数量/型号] 上测试 AMP、梯度累积和 [DDP/ZeRO]，总吞吐由 [A]
  提升至 [B] tokens/s，峰值显存为 [C] GB；通过 profiler 定位 [真实瓶颈]。
```

如果只完成基础版，可写前两条，删除所有未运行的多卡和性能数字。如果实现了 Mini-MoE，应同时报告 total/active parameters、expert utilization 和吞吐，不能只写“加入 MoE 提升性能”。

**面试官最可能核对**：Attention shape 与 mask、LoRA/模型参数量计算、loss shift、为何采用 Pre-Norm、测试如何证明实现正确、吞吐测量是否同步 CUDA。

### 9.5 项目 2：Qwen 微调与对齐简历模板

**建议项目名称**：`基于 Qwen 的领域指令微调、偏好对齐与量化部署`

**适合写入的前提**：至少使用自定义数据完成 QLoRA 与独立评测。只有跑通官方样例时，应写“复现 QLoRA 基线”，不能写“构建领域模型”。

**技术栈候选**：Qwen `[准确版本]`、LLaMA-Factory、Transformers、PEFT、bitsandbytes、TRL、DeepSpeed、vLLM。

```text
基于 Qwen 的领域指令微调、偏好对齐与量化部署｜个人项目｜[时间]
技术栈：Qwen [版本]、LLaMA-Factory、PEFT、[TRL/DeepSpeed]、vLLM
- 面向 [领域与任务] 构建并清洗 [N] 条指令数据，完成去重、长度/来源统计、
  assistant-only label 检查和 train/validation/test 划分，建立 [M] 条独立测试集。
- 使用 NF4 QLoRA 对 [模型规模] 基座完成 SFT，[LoRA rank/target modules] 由
  [显存或消融依据] 确定；在 [指标] 上由 Base 的 [A] 提升至 [B]，并分析
  [事实错误/格式错误/拒答等] bad cases。
- [实际完成 DPO/GRPO 时保留] 构建 [N] 对偏好数据并完成 [DPO/GRPO]，对比
  SFT 基线的 [胜率/KL/正确率]；最终用 [BF16/AWQ/GPTQ] 部署，在 [负载口径]
  下 P95 TTFT/TPOT 为 [A/B]，量化后 [真实质量变化]。
```

如果做了模型合并，应写清“将 LoRA 合并到非量化基座并验证合并前后输出”；如果只动态加载 adapter，就不要写 merged model。若没有运行 PPO/GRPO，论文阅读不能写成“实现强化学习对齐”。

**面试官最可能核对**：数据是否泄漏、chat template 与 label mask、LoRA 参数量、NF4 权重精度与计算精度、DPO/GRPO 目标、量化前后是否使用相同解码配置。

### 9.6 项目 3：企业知识库 RAG 简历模板

**建议项目名称**：`带分层评测与可追溯引用的企业知识库 RAG`

**适合写入的前提**：至少有自定义文档、带 gold evidence 的测试集和检索指标。只有聊天页面而没有测试集时，不适合声称“提升检索准确率”。

**技术栈候选**：Python、BM25、Embedding、HNSW/向量数据库、Reranker、RRF、FastAPI、[LlamaIndex/LangChain]。

```text
带分层评测与可追溯引用的企业知识库 RAG｜个人项目｜[时间]
技术栈：[解析组件]、BM25、[Embedding 模型]、[向量库/HNSW]、[Reranker]、FastAPI
- 处理 [N 篇/页、文档类型]，实现结构化解析、语义切分、元数据/权限保存和
  增量索引；构建 [M] 条带 gold evidence 的测试集，覆盖 [专名/表格/多跳等]。
- 对比 BM25、dense、RRF hybrid 与 reranker，通过 [具体改动] 将 Recall@[K]
  从 [A] 提升至 [B]、MRR 从 [C] 提升至 [D]；将错误分为解析、召回、重排和生成层。
- 实现带 chunk/page 引用的生成 API、文档版本/删除传播与 Prompt Injection 测试；
  faithfulness 为 [A]，并发 [C] 下 P95 为 [B] ms，主要未解决问题为 [真实局限]。
```

若完成 GraphRAG 或 embedding hard-negative 训练，应该说明它改善的是哪个问题子集及带来的构图/训练成本。不能把“接入向量数据库”写成算法创新，也不要把 faithfulness 与答案正确率混为一个指标。

**面试官最可能核对**：chunk size/top-k 的依据、BM25 与 dense 为什么互补、HNSW 参数、测试集如何标注、召回正确但回答错误如何定位、文档如何彻底删除、如何防越权和注入。

### 9.7 项目 4：LangGraph Agent 简历模板

**建议项目名称**：`具备状态恢复、工具安全与轨迹评测的研究 Agent`

**适合写入的前提**：至少有多个真实工具、显式状态、终止/重试逻辑和任务测试集。单次成功演示不能支撑“高可靠 Agent”。

**技术栈候选**：LangGraph、Pydantic/JSON Schema、MCP、FastAPI、向量检索、SQLite/PostgreSQL、Tracing。

```text
具备状态恢复、工具安全与轨迹评测的研究 Agent｜个人项目｜[时间]
技术栈：LangGraph、[MCP]、Pydantic、[存储/检索]、[Tracing 工具]
- 将 planner、retriever、tool executor、verifier 和 human-in-the-loop 建模为显式
  状态图，设计 [关键 state 字段]、checkpoint、预算/循环终止和结构化工具错误。
- 接入 [N] 个实际工具，实现 schema 校验、超时重试、最小权限、幂等键和 [MCP
  adapter/沙箱]；通过重复调用、Prompt Injection、进程中断和并发串线故障注入。
- 构建 [N] 个 Task、每项运行 [K] 次 Trial，分别用 Outcome/Transcript grader 评测；
  pass@1 为 [A]、pass^k 为 [B]，相较普通 RAG/固定 workflow 的 [真实差异] 为 [C]。
```

没有实现 MCP 时直接删除，不要把普通 function calling 写成 MCP；没有做真实写操作时，可以写只读 sandbox，但不要声称实现事务回滚。模型自称“完成”不能作为成功率标签，必须由环境状态、规则或人工 gold 判断。

**面试官最可能核对**：为什么需要 Agent 而不是 workflow、状态如何持久化、如何阻止无限循环、MCP 与 function calling/A2A 的边界、并发结果乱序如何合并、pass@k 与 pass^k 的区别。

### 9.8 项目 5：多模态问答简历模板

**建议项目名称**：`基于 InternVL/Qwen-VL 的领域多模态问答与视觉反事实评测`

**适合写入的前提**：至少完成自定义图文数据、base baseline、一个微调方案和独立测试。只调用官方单图推理接口时不能写“完成多模态微调”。

**技术栈候选**：InternVL 或 Qwen-VL `[准确版本]`、PyTorch、Transformers、PEFT、DeepSpeed、[Gradio/FastAPI]。

```text
基于 [InternVL/Qwen-VL 版本] 的领域多模态问答｜个人项目｜[时间]
技术栈：[模型准确版本]、PyTorch、PEFT、[DeepSpeed]、[FastAPI/Gradio]
- 面向 [票据/图表/OCR/设备面板等] 清洗并构造 [N] 条图文数据，检查坏图、
  图文错配、答案可见性和数据泄漏，建立 [M] 条独立分桶测试集。
- 对比 projector-only、LLM LoRA 与 [实际解冻策略]，记录 trainable parameters、
  视觉 token、峰值显存和各分桶指标；[核心指标] 从 Base 的 [A] 提升至 [B]。
- 设计遮图、换图、OCR 冲突和 [Grounding/高分辨率] 反事实测试，量化视觉幻觉为
  [真实指标]；部署图片问答 API，在 [图片尺寸/并发/GPU] 下 P95 为 [实测值]。
```

如果完成多模态 RAG，应补充跨模态 Recall@K 和 `doc_id/page/bbox` 引用；如果没有 bbox 标注，不要写 Grounding 指标。冻结策略必须与真实可训练参数统计一致。

**面试官最可能核对**：视觉 token 如何进入 LLM、projector 的输入输出 shape、为什么冻结/解冻某模块、高分辨率为何增加显存、反事实测试如何证明模型使用了图像。

### 9.9 强弱描述对照

| 较弱描述 | 更可信的描述方式 |
|---|---|
| 熟练掌握 Transformer | 从零实现 `[模块]`，通过 `[测试]` 与 PyTorch SDPA 数值对齐 |
| 使用 QLoRA 提升模型效果 | 在 `[N]` 条独立测试集上，将 `[指标]` 从 `[A]` 提升到 `[B]`，同时记录 `[回退项]` |
| 搭建高性能 RAG | 对比 BM25/dense/RRF/reranker，在固定 gold 集上报告 Recall@K、faithfulness 和 P95 |
| 使用 LangGraph 实现智能 Agent | 将 `[节点]` 状态化，实现 checkpoint、幂等重试和循环检测，并以环境 grader 评测 `[N]` 个任务 |
| 多卡训练大模型 | 在 `[GPU×数量]`、固定 global batch 下比较 `[DDP/ZeRO]`，报告显存、吞吐和扩展效率 |
| 大幅减少幻觉 | 在明确定义的 `[幻觉测试集]` 上将 `[指标]` 从 `[A]` 降至 `[B]`；仍存在 `[局限]` |

避免使用“精通、深入掌握、业界领先、显著提升、高并发、海量数据”等无法由后续内容证明的词。对于个人项目，“独立完成”可以写，但面试时仍要说清哪些代码来自框架、哪些模块由你实现。

### 9.10 GitHub、简历与面试三者要一致

简历写出的每个项目，仓库至少应能找到以下材料：

```text
README.md                 项目目标、架构、复现命令、结果和局限
configs/                  实际运行配置，不只放示例配置
requirements/lock/env     依赖、CUDA、GPU 与 commit/revision
tests/                    正确性测试和关键故障测试
reports/metrics.json      机器可读的真实指标
reports/experiments.csv   基线与消融原始结果
reports/bad_cases.*       典型错误、归因和修复情况
assets/                   结构图、曲线或 Demo 截图
```

面试前为每条简历 bullet 准备“四个定位”：代码在哪里、配置在哪里、指标在哪里、最大的失败样本在哪里。若面试官问到未完成项，直接说明“没有做过，目前只完成到 `[真实边界]`”，再给出验证计划；诚实的边界通常比虚构完整链路更可信。

### 9.11 投递前的简历自检

- [ ] 简历只保留 2–3 个与目标岗位最相关、完成度最高的项目。
- [ ] 每个项目第一条能说明任务与本人核心实现，不是框架名称列表。
- [ ] 每个指标都有测试集规模、基线和结果文件；性能指标还有硬件与负载口径。
- [ ] “实现、设计、优化、部署、复现”这些动词均与实际工作范围一致。
- [ ] 未完成的 GRPO、GraphRAG、MCP、Grounding、多卡实验已从简历模板中删除。
- [ ] 项目时间、模型版本、数据规模、GPU 和指标在简历、README、面试回答中一致。
- [ ] 每个项目能讲一个失败实验、一次排障和一个仍未解决的局限。
- [ ] 仓库不包含密钥、个人数据、无权公开的数据或体积过大的 checkpoint。

---

## 10. 最终发布检查清单

### 代码与复现

- [ ] 新机器从 README 能创建环境并跑 smoke test。
- [ ] 外部仓库、模型和数据都有版本/commit/revision。
- [ ] 配置、随机种子、GPU、训练预算和解码参数已记录。
- [ ] 密钥、个人信息、checkpoint 未误提交。
- [ ] 核心逻辑有单测；失败路径有故障注入。

### 评测与结论

- [ ] 有明确基线，实验只改变目标变量。
- [ ] 测试集与训练集无明显重复/泄漏。
- [ ] 保存逐样本结果，不只保留平均分。
- [ ] 报告样本数、指标定义、局限和失败案例。
- [ ] 延迟/吞吐注明并发、长度、GPU 和统计口径。

### 简历与面试

- [ ] 每个简历动词与本人实际工作一致。
- [ ] 每个数字能在日志中定位。
- [ ] 能白板画架构和关键张量形状。
- [ ] 每个主项目有一次真实失败与排查故事。
- [ ] 能回答“为什么不用更简单方案”和“扩大十倍怎么办”。

## 11. 官方资料

- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)
- [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)、[PEFT](https://huggingface.co/docs/peft/index)、[TRL](https://huggingface.co/docs/trl/index)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [LangGraph](https://github.com/langchain-ai/langgraph)、[LlamaIndex](https://docs.llamaindex.ai/)、[AutoGen](https://microsoft.github.io/autogen/stable/)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)、[vLLM](https://docs.vllm.ai/)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)
- [MCP 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic)、[A2A v0.3.0](https://a2a-protocol.org/v0.3.0/specification/)

> 访问日期：2026-07-14。开始项目时再次读取对应 commit 的安装、数据和迁移说明；若官方接口变化，在 README 记录差异和解决办法，这本身也是工程能力证据。
