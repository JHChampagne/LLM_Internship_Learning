# 大模型算法工程技术学习指南

> 面向大模型算法实习的“原理—代码—实验—表达”教材。更新基线：2026-07-14。  
> 配套：[面试题库](./02_Interview_QA.md)｜[实战项目手册](./03_Practical_Projects.md)

## 目录

1. Python、PyTorch 与工程基础
2. Transformer：从公式到代码
3. 数据、Tokenizer 与预训练
4. SFT、LoRA、QLoRA 与偏好对齐
5. 训练稳定性、显存和性能优化
6. 分布式训练：DDP、DeepSpeed、Megatron
7. Prompt、推理与评测
8. RAG：检索增强生成
9. Agent、工具调用与记忆
10. 多模态大模型
11. 世界模型、强化学习与论文复现
12. 8–12 周学习路线与速查附录

**先修知识**：Python 基础、线性代数（矩阵乘法、概率与导数）和基本 Linux 命令。若缺少某项，可先跑通 §1 的最小训练循环，再回补数学细节。  
**学习目标**：能解释关键公式、写出最小实现、运行框架流程、设计公平实验，并用证据诊断常见训练和应用故障。

## 0. 如何使用本指南

学完一个主题，不应只会复述定义，而应能完成四件事：

1. **解释**：不用术语堆砌，说明它解决什么问题、核心假设是什么。
2. **推导**：写出关键公式，说明张量形状、时间/空间复杂度。
3. **操作**：能运行最小代码，能解释重要参数和日志。
4. **诊断**：面对 OOM、NaN、低吞吐、效果下降，能提出排查顺序。

建议为每次实验保存：`config.yaml`、代码提交号、环境锁文件、随机种子、原始日志、指标 JSON、错误样本和结论。面试中的“做过”应由这些证据支撑。

### 0.1 默认环境

- Linux、Python 3.11+、NVIDIA GPU；训练前执行 `nvidia-smi` 确认驱动和可用显存。
- PyTorch 为主。TensorFlow/JAX 只要求理解执行模型和生态差异。
- 不盲目锁死一组 CUDA 版本：先根据云主机驱动，在 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 生成命令，再安装其余依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# 按官方安装页安装匹配 CUDA 的 torch 后：
pip install transformers datasets accelerate peft trl sentence-transformers
pip install jupyter matplotlib pandas scikit-learn tensorboard
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

生产实验应将实际解析后的依赖写入 `requirements-lock.txt`：

```bash
python -m pip freeze > requirements-lock.txt
```

### 0.2 一次实验的最小记录

```text
runs/exp_001/
├── config.yaml          # 超参数与数据版本
├── env.txt              # pip freeze、GPU、驱动
├── train.log            # loss、lr、tokens/s、显存
├── metrics.json         # 可机器读取的评测结果
├── samples.jsonl        # 典型成功/失败样本
└── conclusion.md        # 假设、结果、局限、下一步
```

---

## 1. Python、PyTorch 与工程基础

### 1.1 Python 在大模型工程中的角色

Python 负责数据、训练编排、评测和服务胶水；高性能算子通常由 CUDA/C++/Triton 实现。面试所说“Python 扎实”通常包括：容器和迭代器、装饰器/上下文管理器、类型标注、异常处理、并发基础、包管理、性能分析和可测试代码。

需要能解释以下工程习惯：

- 配置与代码分离；密钥从环境变量读取，不写入仓库。
- 数据读取使用生成器/流式 Dataset，避免一次性载入内存。
- 训练函数显式接收配置，固定随机种子，但理解 GPU 算子并非总是完全确定。
- 用日志记录结构化指标，不用散落的 `print` 代替实验系统。

```python
from dataclasses import dataclass
import os, random
import numpy as np
import torch

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 4

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

cfg = TrainConfig()
seed_everything(cfg.seed)
api_key = os.environ.get("MODEL_API_KEY")
```

### 1.2 张量、自动微分和训练循环

Tensor 是带设备和数据类型的多维数组。`requires_grad=True` 时，PyTorch 动态记录计算图；`loss.backward()` 根据链式法则把梯度累积到叶子张量的 `.grad`。因此每个更新周期必须清梯度。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(1024, 32)
y = (x[:, :1] * 0.7 + torch.randn(1024, 1) * 0.1)
loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.MSELoss()

for epoch in range(3):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(epoch, float(loss), float(grad_norm))
```

关键点：

- `model.train()`/`eval()` 影响 Dropout、BatchNorm；推理还应使用 `torch.inference_mode()`。
- AdamW 将权重衰减与梯度更新解耦；通常不对 bias 和归一化参数做 weight decay。
- `set_to_none=True` 通常更省内存；但代码若假设 `.grad` 总是 Tensor，需特别处理。
- DataLoader 性能看 `num_workers`、`pin_memory`、数据预处理和存储吞吐，不是越大越好。

### 1.3 PyTorch、TensorFlow、JAX 怎么比较

| 框架 | 核心特点 | 常见场景 | 面试表达 |
|---|---|---|---|
| PyTorch | eager-first、生态广、研究到部署衔接强 | LLM/VLM 训练主流 | 动态调试方便，`torch.compile` 可做图优化 |
| TensorFlow | Keras、TF Serving、成熟生产生态 | 既有工业系统 | 图执行和部署工具完整 |
| JAX | 函数式变换，`jit/vmap/pmap`，XLA | 大规模研究、TPU | 纯函数与显式 PRNG，编译和分片能力强 |

不要回答“一个动态图一个静态图”就结束：现代框架都能在 eager 与编译图之间工作，真正区别在编程模型、分布式抽象、生态和团队成本。

### 1.4 Linux、Git、Hugging Face

常用排障命令：

```bash
nvidia-smi
watch -n 1 nvidia-smi
df -h && free -h
ps -ef | grep python
ss -lntp
du -sh checkpoints/*
git status && git diff
git log --oneline -10
```

Hugging Face 的关键抽象：

- `transformers`：模型、Tokenizer、训练/生成接口。
- `datasets`：Arrow 数据集、`map`、streaming。
- `accelerate`：设备放置与分布式启动。
- `peft`：LoRA 等参数高效微调。
- `trl`：SFT、DPO、GRPO/PPO 等后训练组件。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"  # 若仓库名称变化，以模型卡为准
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
messages = [{"role": "user", "content": "用一句话解释反向传播。"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

`chat_template` 很重要：同一语义若使用错误角色标记，效果可能明显下降。生成时要区分输入 token 与新增 token。

### 1.5 数学与机器学习高频基础

#### Softmax、交叉熵与 KL

令 `p=softmax(z)`，其 Jacobian 为：

$$
\frac{\partial p_i}{\partial z_j}=p_i(\mathbf 1[i=j]-p_j)
$$

若标签是 one-hot `y`，交叉熵 `L=-\sum_i y_i\log p_i`，则 logits 梯度化简为 `∂L/∂z=p-y`。这解释了为什么“softmax + cross entropy”通常实现为一个数值稳定算子：直接先算概率再取 `log` 容易上溢/下溢，工程上应使用 `log_softmax` 或 `cross_entropy`。

- NLL 是对正确类别负对数似然；one-hot 分类时与交叉熵等价。
- `H(q,p)=H(q)+D_KL(q||p)`；固定真实分布 `q` 时，最小化交叉熵等价于最小化前向 KL。
- 前向 KL 对漏掉 `q` 的概率质量惩罚大，常呈 mass-covering；反向 KL 在多峰分布中更可能 mode-seeking。这是倾向，不应脱离具体分布绝对化。

```python
import torch

z = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
y = torch.tensor([1])
loss = torch.nn.functional.cross_entropy(z, y)
loss.backward()
expected = torch.softmax(z.detach(), -1) - torch.nn.functional.one_hot(y, 3)
torch.testing.assert_close(z.grad, expected)
```

#### 正则化、优化器与 warmup

- L1 加 `λ||w||₁`，在零点不可导但可用次梯度，倾向稀疏；L2 加 `λ||w||²/2`，梯度增加 `λw`，使大权重受更强惩罚。
- SGD 中 L2 与 weight decay 可等价；带动量/自适应缩放的 Adam 中并不严格等价。AdamW 将衰减从 loss 梯度中解耦，先按 Adam 更新，再直接缩放参数。
- warmup 从小学习率升至目标值，缓解训练初期随机参数、未稳定统计和大梯度造成的破坏；之后常接 cosine 或 linear decay。warmup 步数应按总 optimizer steps，而非 micro-steps 计算。
- bias、LayerNorm/RMSNorm scale 通常不做 weight decay，但应以实验验证，不是数学定律。

#### 指标、过拟合与梯度异常

二分类 `precision=TP/(TP+FP)`、`recall=TP/(TP+FN)`、`F1=2PR/(P+R)`。ROC-AUC 可理解为随机正样本得分高于随机负样本的概率，阈值无关，但类别极不平衡时 PR-AUC 往往更有解释力。多分类必须说明 macro/micro/weighted 平均。

诊断顺序：

1. 训练好、验证差：先查数据泄漏、划分与分布漂移，再考虑正则化、早停、数据增强和降低容量。
2. 梯度爆炸：记录 global/per-layer norm，检查学习率、异常长样本、loss mask、低精度溢出；再使用 warmup、裁剪或稳定初始化。
3. 梯度消失：检查饱和激活、过深链路与错误 detach；残差、归一化和合适初始化通常比盲目增大学习率更根本。
4. loss 为 NaN：定位第一个异常 step 和第一个非有限张量，不要只在末尾过滤 NaN。

**证明理解**：手推 `softmax+CE` 梯度；在同一数据上画 train/valid loss、learning rate、grad norm；解释一次阈值变化为何改变 F1 却不改变 ROC-AUC。

**自检**：能否解释为什么梯度会累积？为什么训练和验证要切换模式？如何记录一次可复现实验？

---

## 2. Transformer：从公式到代码

### 2.1 Self-Attention

输入 `X∈R^(B×T×d_model)` 经线性投影得到 Q、K、V。缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

`M` 是 mask：Causal LM 把未来位置设为负无穷；Padding mask 屏蔽补齐 token。除以 `sqrt(d_k)` 是为了控制点积方差，避免 softmax 过早饱和、梯度过小。

```python
import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z):
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = map(split, (q, k, v))                    # [B,H,T,Dh]
        score = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        score = score.masked_fill(mask, torch.finfo(score.dtype).min)
        prob = torch.softmax(score.float(), dim=-1).to(q.dtype)
        y = (prob @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)
```

时间复杂度主要是 `O(T²d)`，注意力矩阵显存约为 `O(BHT²)`；长上下文优化会从 FlashAttention、稀疏/滑窗注意力、GQA/MQA、KV Cache 压缩等方向入手。

### 2.2 Multi-Head、残差、归一化与 FFN

多头使不同子空间能关注不同关系。一个现代 Decoder Block 通常是 Pre-Norm：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

- Pre-Norm 比早期 Post-Norm 更利于深层训练。
- RMSNorm 只按均方根缩放，不减均值，计算更简单。
- LLaMA/Qwen 常见 SwiGLU：`down(silu(gate(x)) * up(x))`，参数量和表达力与普通两层 FFN 不同。
- 残差提供梯度高速通道，但残差尺度、初始化仍会影响深层稳定性。

### 2.3 位置编码

- Learned absolute：每个位置一个可学习向量，简单但外推弱。
- Sinusoidal：固定正余弦，具有相对位置信息结构。
- RoPE：对 Q/K 分组旋转，让点积自然包含相对位置信息；现代 Decoder LLM 常用。
- ALiBi：注意力分数加入与距离相关的线性偏置。

上下文扩展不是简单修改 `max_position_embeddings`：需要考虑训练分布、RoPE scaling、注意力显存和长文本评测。

### 2.4 Encoder、Decoder 与模型家族

| 模型 | 结构 | 训练目标 | 典型用途 |
|---|---|---|---|
| BERT | Encoder-only | Masked LM | 表示、分类、抽取 |
| GPT/LLaMA/Qwen | Decoder-only | Next-token prediction | 生成、对话、Agent |
| T5 | Encoder-Decoder | Text-to-text denoising | 条件生成、转换任务 |

LLaMA 类模型的重要工程特征常包括 RoPE、RMSNorm、SwiGLU、GQA；Qwen 系列在此基础上扩展 tokenizer、长上下文、工具调用和多模态能力。具体结构必须查看对应模型的 `config.json` 和模型卡，不能把家族特征当成每个版本的绝对事实。

### 2.5 KV Cache

自回归生成第 `t` 个 token 时，过去 token 的 K/V 不变，可缓存后复用。于是每步无需重新计算历史 K/V，但 Cache 显存随 `batch × layers × sequence × kv_heads × head_dim × 2 × bytes` 增长。GQA/MQA 通过减少 KV heads 降低 Cache。

排障：吞吐低但 GPU 利用率不高，可能是小 batch、逐请求调度或 CPU/tokenizer 瓶颈；长上下文 OOM 可能主要来自 KV Cache，而不是模型权重。

### 2.6 MHA、MQA、GQA、MLA 与 FlashAttention

设 query heads 数为 `Hq`，KV heads 数为 `Hkv`：

| 结构 | `Hkv` | 优点 | 代价/注意 |
|---|---:|---|---|
| MHA | `Hq` | 每个头有独立 K/V，表达直接 | KV Cache 最大 |
| GQA | `1 < Hkv < Hq` | 在质量与缓存/带宽间折中 | query heads 按组共享 KV |
| MQA | `1` | KV Cache 最小、decode 读带宽少 | 共享过强可能影响质量 |
| MLA | 潜在低维表示 | 压缩需要缓存的 KV 表示 | 不是“更多共享头”；实现与 RoPE 解耦需按论文分析 |

GQA 的 KV Cache 元素量仍可用 `2×B×L×T×Hkv×Dh` 粗估。把 MHA 改为 GQA 时，训练中常用 mean pooling 将一组旧 K/V heads 初始化成一个 KV head；不能只改 `num_key_value_heads` 而不处理权重形状。

```python
def repeat_kv(kv, groups: int):
    # kv: [B, Hkv, T, Dh] -> [B, Hq, T, Dh]
    return kv.repeat_interleave(groups, dim=1)

# PyTorch SDPA 会按设备、dtype、mask 等条件选择可用后端；不要承诺必然走 Flash。
y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

FlashAttention 不改变精确 attention 的数学目标，而是通过 tiling、online softmax 和减少 HBM 往返读写降低 IO 与中间矩阵存储。普通实现会物化 `T×T` scores/probabilities；FlashAttention 分块维护每行的最大值 `m`、归一化和 `l` 与部分输出。若新块最大值变大，旧累积项需乘 `exp(m_old-m_new)` 重新缩放。因此它是 IO-aware 的精确算法，不等于稀疏注意力，也不能把理论 `O(T²d)` 计算量说成线性。

验证时同时比较：数值误差、峰值显存、prefill 延迟、不同序列长度；先 warm up，再同步 CUDA，固定 dtype/mask/dropout。参考 [FlashAttention](https://arxiv.org/abs/2205.14135) 与 [FlashAttention-2](https://arxiv.org/abs/2307.08691)。

### 2.7 MoE、Scaling Law 与具体模型版本

稀疏 MoE 将 FFN 替换为多个 experts。router 对 token `x` 产生 logits，选择 top-k experts：

$$p=\operatorname{softmax}(W_rx),\quad y=\sum_{i\in TopK(p)}p_iE_i(x)$$

总参数可很大，但每 token 只激活少数 experts；代价是路由、capacity、token dispatch/combine 和跨卡 All-to-All。负载不均会让热门 expert OOM、尾部延迟升高、其他 expert 学不到。常见方法是 auxiliary load-balancing loss、expert capacity/drop token，以及只用于路由决策的动态 bias。后者不能简化成“完全没有均衡机制”。

Dense 与 MoE 的公平比较必须固定至少一个口径：active parameters、训练 FLOPs、总参数、训练 token 或服务预算；只说“MoE 参数多但计算少”不够。MoE 在小 batch、慢互联或低并发下未必更快。

Scaling law 描述 loss 与参数量 `N`、数据量 `D`、计算量 `C` 的经验幂律关系，用于给定预算下分配模型与数据。它是特定数据、架构和训练区间的经验外推，不是越大必然按同一斜率变好；数据质量、推理时计算和 MoE 会改变口径。

**版本化模型卡片（访问日期 2026-07-14）**：

| 具体版本 | 可安全陈述的结构点 | 不应泛化的结论 |
|---|---|---|
| LLaMA 1（2023 论文） | Decoder-only、Pre-Norm/RMSNorm、SwiGLU、RoPE；不同规模头数不同 | 不能据此断言所有后续 Llama 都使用相同 attention/上下文配置 |
| Qwen3 初版（2025-04/05 官方仓库与报告） | 同系列同时发布 dense/MoE 规格，强调 thinking/non-thinking 模式与工具能力 | “Qwen 全系列都是 MoE”或“每个 Qwen3 checkpoint 都同结构”均错误 |
| DeepSeek-V3（arXiv:2412.19437） | 671B 总参数、每 token 激活约 37B，MLA、DeepSeekMoE、MTP；报告 auxiliary-loss-free load balancing 策略 | 不要把 V3 特征套到所有 DeepSeek 模型；性能数字需连同评测配置引用 |

面试前读取目标 checkpoint 的 `config.json`：`hidden_size`、层数、`num_attention_heads`、`num_key_value_heads`、RoPE、MoE experts/top-k、词表与最大位置。结构题先说版本，再比较。参考 [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)、[Qwen3 官方仓库](https://github.com/QwenLM/Qwen3) 与 [DeepSeek-V3 报告](https://arxiv.org/abs/2412.19437)。

**实验**：实现上面的 Attention，与 `torch.nn.functional.scaled_dot_product_attention` 在关闭 dropout 时对齐输出；测量 T 从 128 到 2048 的耗时/峰值显存。

**面试表达**：“我会先给公式和形状，再解释缩放、mask、多头的作用，最后落到 O(T²) 瓶颈和 FlashAttention/KV Cache 的工程优化。”

---

## 3. 数据、Tokenizer 与预训练

### 3.1 数据流水线

典型流程：采集 → 去重 → 质量过滤 → 隐私/安全清洗 → 文档切分 → Tokenize → packing → train/validation/test 划分 → 版本化。

必须防止数据泄漏：相同文档或近重复样本不能跨训练集和测试集；面向时间预测的任务按时间切分。质量不是只看数量，还要检查长度分布、语言/领域比例、模板分布、拒答比例和答案正确性。

### 3.2 Tokenizer

BPE、WordPiece、Unigram 都在子词粒度平衡词表大小与序列长度。Tokenizer 的词表和切分规则是模型的一部分，不能随意替换。新增 special token 后通常需要 resize embedding，并考虑这些新向量如何训练。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
ds = load_dataset("json", data_files={"train": "train.jsonl"})["train"]

def encode(batch):
    texts = [tok.apply_chat_template(x, tokenize=False) for x in batch["messages"]]
    return tok(texts, truncation=True, max_length=2048)

tokenized = ds.map(encode, batched=True, remove_columns=ds.column_names)
print(tokenized)
```

### 3.3 预训练目标

Causal LM 最小化：

$$L=-\sum_t \log p_\theta(x_t|x_{<t})$$

输入和标签通常错位一位。Padding、用户提示等不希望计入 loss 的位置用 `-100` 屏蔽。Packing 可把多个短样本拼进定长序列提高 token 利用率，但需要正确处理样本边界和 mask。

预训练看 token-level loss/perplexity、下游基准和训练稳定性；不能仅用训练 loss 判断通用能力。

**自检**：为什么按样本随机切分可能泄漏？新增 token 为什么要 resize embedding？packing 如何提升 MFU？

---

## 4. SFT、LoRA、QLoRA 与偏好对齐

### 4.1 SFT

Supervised Fine-Tuning 用高质量“输入—理想输出”继续做条件语言建模。核心不是“让模型记住答案”，而是调整行为分布和任务格式。

关键参数：

- learning rate：全参微调通常比 LoRA 更小；过大会灾难性遗忘。
- effective batch：`micro_batch × grad_accum × data_parallel_world_size`。
- sequence length：影响截断率与显存；先统计数据长度分布。
- epoch：小数据反复训练易过拟合；看验证集和能力回退。
- loss mask：只训练 assistant 回复，还是连 prompt 一起训练，必须明确。

### 4.2 LoRA

对冻结权重 `W∈R^(d_out×d_in)`，学习低秩增量：

$$W'=W+\Delta W,\quad \Delta W=\frac{\alpha}{r}BA$$

其中 `A∈R^(r×d_in)`、`B∈R^(d_out×r)`，新增参数为 `r(d_in+d_out)`，远小于 `d_in d_out`。`r` 控制容量，`alpha/r` 控制缩放，target modules 决定注入位置。

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype="auto", device_map="auto", trust_remote_code=True
)
cfg = LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(base, cfg)
model.print_trainable_parameters()
```

### 4.3 QLoRA

QLoRA 将冻结基座以 4-bit（常用 NF4）加载，在其上训练 LoRA；计算通常仍用 BF16/FP16。量化权重并非普通可训练参数，所以“4-bit 全参训练”与 QLoRA 不是一回事。

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", quantization_config=bnb, device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
```

以当前官方文档为准：bitsandbytes/Transformers 的量化接口和硬件支持持续变化；4/8-bit 训练通常只更新额外参数。参考 [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) 与 [PEFT quantization](https://huggingface.co/docs/peft/main/developer_guides/quantization)。

### 4.4 DPO、RLHF、PPO、Agentic RL、OPD

**RLHF 经典链路**：SFT → 偏好数据 → Reward Model → PPO 优化 policy，并以 KL 约束避免偏离参考模型。优点是目标灵活；难点是系统复杂、reward hacking 和训练不稳定。

**DPO** 不显式训练 Reward Model，而从 Bradley–Terry 偏好模型推导出直接的二分类式目标。对 `(x, y_w, y_l)`，直观上提高 chosen 相对 rejected 的对数概率差，同时通过参考模型隐式约束：

$$
L_{DPO}=-\log\sigma\{\beta[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))-(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))]\}
$$

`β` 控制偏好优化与贴近参考策略的权衡。数据若偏好标签噪声大、长度偏置强，DPO 也会学习这些偏差。

**Agentic RL** 将模型置于可交互环境，奖励来自多步任务完成、工具调用或验证器；难点包括长时序 credit assignment、稀疏奖励、环境非确定性和昂贵 rollout。

**On-Policy Distillation (OPD)** 用教师对当前学生策略产生的 on-policy 轨迹提供软目标/反馈，缓解离线蒸馏中的分布偏移。具体名称和目标在不同论文中不完全统一，面试时应先说明所指论文，不能把缩写当成固定算法。

TRL 当前覆盖 SFT、DPO、Reward Modeling、GRPO/PPO 等，接口以 [官方文档](https://huggingface.co/docs/trl/index) 为准。

### 4.5 如何设计对照实验

至少保持相同基座、数据划分、最大长度、生成参数和评测脚本，对比：Base、SFT、LoRA 不同 rank、QLoRA、DPO。除平均分外，保存每类错误：格式错误、事实错误、拒答过度、指令遗漏、长度偏置。

### 4.6 Prompt/Prefix、Adapter、AdaLoRA 与蒸馏

| 方法 | 训练对象 | 插入位置/直觉 | 推理影响 |
|---|---|---|---|
| Prompt Tuning | 少量虚拟 input embeddings | 在输入端学习 soft prompt | 增加少量输入 token |
| Prefix Tuning | 各层可学习 K/V prefix | 给每层 attention 提供条件 | 每层 KV 变长 |
| Adapter | 瓶颈 MLP 残差模块 | 在 block 内学习任务变换 | 增加串行算子延迟 |
| LoRA | 低秩权重增量 | 修改线性变换的有效权重 | 可合并，或动态挂载 |
| AdaLoRA | 动态分配 rank/预算 | 依据重要性在模块间重分配容量 | 训练流程更复杂 |

知识蒸馏不是简单复制教师最终答案：logit distillation 用温度 `τ` 软化分布并最小化 KL；sequence-level distillation 用教师生成序列做监督；feature distillation 对齐中间表示；on-policy distillation 让教师评价/指导学生当前采样，减轻离线数据分布偏移。高温 KL 常乘 `τ²` 保持梯度量级。蒸馏必须检查教师错误、风格泄漏、学生容量上限和测试集污染。

### 4.7 Reward Model、GRPO、GSPO、DAPO、RLAIF 与信用分配

Reward Model 常用 Bradley–Terry 成对损失：

$$L_{RM}=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$$

绝对 reward 值不可跨 RM 随意比较；重点是排序、校准、长度/格式偏置和分布外鲁棒性。

**GRPO** 对同一 prompt 采样一组回答，以组内 reward 均值/标准差构造相对 advantage，省去 PPO 的独立 value model，再用 clipped ratio 与 KL/参考策略约束更新。若组内 reward 全同，优势接近零；组大小、采样温度、验证器质量和 reward variance 都影响学习，不能只背“比 PPO 省显存”。原始来源是 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。

**GSPO**（arXiv:2507.18071）把 importance ratio 与 clipping 提到序列级，论文目标包括改善长序列及 MoE RL 稳定性；**DAPO**（arXiv:2503.14476）是基于 GRPO 的系统化配方，强调 Clip-Higher、动态采样、token-level policy-gradient loss、overlong reward shaping 等组件。二者不是“GRPO 的同义词”，比较时应写出各自优化粒度和实现版本。前沿方法结论要限定论文与实验环境。

**RLAIF** 用 AI 反馈替代或补充人工偏好，可扩大量级，但把 judge 的偏差、提示敏感性和同源模型偏好带入 reward。应抽样人工校准、交换答案顺序、多 judge/规则验证，并保留原始理由与版本。

**信用分配**回答“最终回报应归因到哪些 token/动作”。结果奖励易获取但稀疏；过程奖励更密集，却需要可靠步骤标注/验证。Agentic RL 还需处理工具副作用、环境随机性、轨迹截断和跨多步延迟回报。使用 GAE/value baseline、组内相对 advantage 或 reward-to-go 都是在降方差/分配信用，不会自动修复错误 reward。

最小 GRPO 伪代码：

```text
for prompt in batch:
    ys = policy.sample(prompt, G)
    rewards = verifier(prompt, ys)
    advantages = normalize_within_group(rewards)
    loss += clipped_policy_loss(ys, advantages, old_policy, reference_policy)
update(policy, loss)
```

**自检**：能否手算某线性层 LoRA 参数量？QLoRA 的权重精度和计算精度是否相同？DPO 为什么仍需要 reference policy？

---

## 5. 训练稳定性、显存和性能优化

### 5.1 显存由什么构成

训练显存近似由：参数、梯度、优化器状态、激活、临时 buffer、通信 bucket 和碎片组成。以 AdamW 全参混合精度训练为例，每参数可能需要低精度参数、梯度、FP32 master weight 及两份 FP32 动量；不能只用“参数量 × 2 bytes”估算训练显存。

推理主要是权重、KV Cache、激活/临时 workspace。量化主要降低权重，长上下文/大并发时 KV Cache 仍可能主导。

### 5.2 AMP、梯度累积与裁剪

```python
scaler = torch.amp.GradScaler("cuda")
optimizer.zero_grad(set_to_none=True)
for step, (x, y) in enumerate(loader):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(x, labels=y).loss / grad_accum
    scaler.scale(loss).backward()
    if (step + 1) % grad_accum == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

- FP16 范围小，GradScaler 防止梯度下溢；BF16 指数范围大，通常不需要 scaling，但精度位更少。
- 裁剪应在 `unscale_` 后进行。
- 累积期间 loss 要除以累积步数；最后不足整组的 batch 要特别处理。
- NaN 排查顺序：定位首次出现 step → 检查数据/label → 学习率和 warmup → 精度与 loss scaling → 梯度 norm → 自定义算子。

参考 [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

### 5.3 Checkpoint 与恢复

完整恢复需保存 model、optimizer、scheduler、scaler、global step、随机数状态、sampler 状态和数据位置。只加载模型权重叫 warm start，不是严格 resume。分布式 checkpoint 还涉及分片布局和 world size 变化。

### 5.4 性能指标

- throughput：tokens/s 或 samples/s；必须说明总吞吐还是单卡。
- latency：首 token 延迟 TTFT、每输出 token 延迟 TPOT、端到端 P50/P95/P99。
- MFU：实际 FLOPs/s 与硬件理论峰值之比，计算口径要一致。
- GPU utilization 高不代表有效吞吐高，可能在做冗余计算。

优化顺序：先 profile，判断是数据、CPU、H2D、计算、显存还是通信瓶颈；再调整 batch/packing、融合算子/FlashAttention、checkpointing、并行策略和通信重叠。

### 5.5 Prefill、Decode、PagedAttention 与调度

推理有两个不同阶段：

- **Prefill**：一次处理全部输入 token，矩阵较大、并行度高，通常更 compute-bound；生成第一个 token，影响 TTFT。
- **Decode**：每轮每请求通常只生成一个 token，读取模型权重和 KV Cache，常更 memory-bandwidth-bound；影响 TPOT/ITL。

`TTFT = 请求到达至首 token`；`TPOT/ITL` 是首 token 后相邻输出 token 的平均/分位延迟；端到端 latency 还包含排队、tokenize、网络与 detokenize。throughput 和 latency 会互相制约，必须同时报告请求长度分布、并发、batch policy、硬件和分位数。

PagedAttention 把每个请求的 KV Cache 划成固定大小逻辑块，按需映射到非连续物理块，减少预留与碎片，并支持共享前缀块。它解决缓存管理，不改变 attention 公式。continuous batching 在 decode step 间动态加入新请求、移除已完成请求，提高 GPU 占用；代价是调度开销与尾延迟权衡。prefix cache 只复用完全相同 token 前缀对应的 KV；模型权重、LoRA、RoPE/位置、cache salt 等不一致时不能错误复用。

### 5.6 量化粒度、格式与推理引擎

量化需说清四个维度：对象（weight/activation/KV）、位宽、粒度（per-tensor/channel/group/token）和校准/训练方式。粒度更细通常误差小但 scale 元数据与 kernel 更复杂。

| 名称 | 核心定位 | 易错点 |
|---|---|---|
| GPTQ | 基于校准样本的逐层/近似二阶权重量化 | 是算法/权重格式生态，不等于任意 INT4 |
| AWQ | 根据激活识别显著权重并保护/缩放后量化 | 依赖代表性校准数据与硬件 kernel |
| GGUF | llama.cpp 生态的模型容器/量化格式 | 不是训练算法；常用于 CPU/边缘本地推理 |
| bitsandbytes NF4 | QLoRA 常用 4-bit 数据类型与加载路径 | 训练通常只更新附加参数 |

部署选型：vLLM 强调高吞吐服务、PagedAttention、continuous batching 与 OpenAI-compatible API；TGI 是 Hugging Face 的文本生成服务栈；TensorRT-LLM 面向 NVIDIA GPU，强调图/算子优化与量化。选型以模型支持、硬件、动态 batching、LoRA、量化、可观测性和运维能力为准，不只比较单条离线速度。

```bash
# 示例按 vLLM 当前文档核对；模型名、dtype、量化格式必须匹配模型卡。
vllm serve Qwen/Qwen3-0.6B --dtype auto --max-model-len 4096
```

验证量化需同时测任务质量、困惑度（适用时）、显存、TTFT/TPOT、吞吐和加载时间。INT4 理论存储下降不保证端到端等比例提速：反量化、kernel、batch、带宽及 KV Cache 都可能成为瓶颈。参考 [vLLM 官方文档](https://docs.vllm.ai/)。

---

## 6. 分布式训练：DDP、DeepSpeed、Megatron

### 6.1 数据并行（DP/DDP）

每个 rank 有完整模型，处理不同数据；反向时 All-Reduce 梯度。DDP 通常一 GPU 一进程，并配 `DistributedSampler`。

```bash
torchrun --standalone --nproc-per-node=4 train.py
```

```python
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

常见坑：每个 rank 用了同一批数据、只在 rank 0 调 backward、条件分支导致 unused parameter、所有 rank 都保存同一文件、某 rank 异常导致其他 rank 卡在 collective。

### 6.2 ZeRO

- ZeRO-1：分片 optimizer states。
- ZeRO-2：再分片 gradients。
- ZeRO-3：再分片 parameters，需要时 All-Gather。
- Offload：把状态卸载到 CPU/NVMe，以速度换显存。

```json
{
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": 8,
  "train_micro_batch_size_per_gpu": 1,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

ZeRO stage 越高不保证越快：模型较小、网络较慢或 bucket 不合适时，额外通信会抵消显存收益。参考 [DeepSpeed ZeRO](https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/zero.md)。

### 6.3 Megatron 并行维度

- TP：矩阵沿维度切到多卡，层内频繁 collective；适合单层放不下一卡。
- PP：按层切 stage，用 micro-batch 填流水线；存在 pipeline bubble。
- DP：复制模型，分数据。
- EP：MoE experts 分到不同卡，核心通信常为 All-to-All。
- CP：长序列上下文维度分片。

并行方案取决于模型大小、节点内 NVLink/NVSwitch、跨节点网络、序列长度和 batch。Megatron Core 当前提供 TP/PP/DP/EP/CP 等组合，参考 [官方仓库](https://github.com/NVIDIA/Megatron-LM)。

### 6.4 计算与通信瓶颈怎么定位

1. 记录单卡基线 tokens/s。
2. 逐步扩展 2/4/8 卡，计算 scaling efficiency：`多卡吞吐/(卡数×单卡吞吐)`。
3. 用 PyTorch Profiler/Nsight 看 kernel gap、collective 时间和数据等待。
4. 检查张量大小、bucket、micro-batch、梯度累积、网络拓扑。
5. 尝试通信计算重叠、增大有效计算粒度、调整并行维度。

**面试表达**：不要只背 ZeRO 三阶段；给出“显存收益—通信代价—适用条件”的权衡。

### 6.5 FSDP、Accelerate、collective 与粗估

PyTorch FSDP 与 ZeRO-3 都可分片参数、梯度和优化器状态；FSDP 是 PyTorch 原生模块包装/`fully_shard` 路径，便于与原生生态组合，DeepSpeed ZeRO 提供自己的 engine、offload 与大量训练系统能力。不能仅凭“stage 对应关系”断言性能相同：预取、wrap 粒度、checkpoint 格式、CPU offload、通信 bucket 和版本都会改变结果。

Accelerate 是统一设备放置、混合精度和多种分布式后端的编排层；它可以启动 DDP/FSDP/DeepSpeed，但本身不是新的并行算法。排障时要能下钻到底层配置。

常见 collective：

- All-Reduce：DDP 汇总梯度；逻辑上 reduce 后每 rank 都有结果。
- Reduce-Scatter + All-Gather：分片梯度/参数常用，可组成 All-Reduce。
- All-Gather：收集参数分片或张量分片。
- All-to-All：MoE token 在 expert ranks 间重分发。
- Send/Recv：流水线 stage 间传激活/梯度。

Ring All-Reduce 每 rank 通信量粗略为 `2(N-1)/N × tensor_bytes`；带宽模型可写 `T≈α×消息轮次+bytes/BW`。小消息常受延迟 `α` 控制，大消息受带宽控制。跨节点慢时应确认 collective 是否跨过低带宽链路。

粗估方法：

- Decoder-only 训练 FLOPs 常用 `≈6×参数量×训练 token 数` 作数量级估算；它忽略 attention 的长序列额外项、MoE active/total 差异、重计算与实现开销。
- 权重显存 `P×bytes`；AdamW 全参训练再分别加梯度、master weights、两份 moments、激活和碎片。必须写出假设，不能把经验“16/18 bytes 每参数”当普适定律。
- 实测记录 `max_memory_allocated/reserved`、tokens/s、collective 时间与 scaling efficiency，用来校正纸面估计。

---

## 7. Prompt、推理与评测

### 7.1 Prompt 与 Context Engineering

Prompt Engineering 设计单次输入的指令、约束、示例和输出格式；Context Engineering 更广，负责在有限 context window 内选择、组织、压缩和更新系统提示、历史、检索证据、工具定义与中间状态。

稳定 Prompt 的结构：角色/目标、输入边界、规则、可用证据、输出 schema、失败策略、少量高质量示例。避免只靠“请认真思考”；应提供可验证步骤和结构化输出。

CoT 是产生中间推理的提示/训练范式；ReAct 在推理与行动之间交替。生产系统不应依赖向用户暴露完整隐式推理，可要求简短依据、引用和可审计工具轨迹。

### 7.2 生成参数

- `temperature` 缩放 logits；越高通常越随机。
- `top_p` 做 nucleus sampling；与 temperature 同时调会耦合。
- `max_new_tokens` 限制新增长度，不等于总上下文。
- 重复惩罚可能损害代码/结构化文本；先用任务评测验证。
- 评测比较时固定解码参数和 chat template；确定性任务常用 greedy/temperature 0。

### 7.3 评测体系

评测分层：

1. 离线能力基准：准确率、Exact Match、F1、pass@k。
2. 任务指标：业务规则通过率、结构合法率、引用正确率。
3. 人工评测：事实、完整、风格、安全；需盲评和清晰 rubric。
4. LLM-as-a-Judge：便宜可扩展，但有位置、长度、自偏好偏差；要校准人工一致性。
5. 在线指标：成功率、延迟、成本、用户反馈与安全事件。

`perplexity=exp(平均 token NLL)`，只适合相同 tokenizer/数据处理下谨慎比较；低 PPL 不等于指令遵循好。评测要报告样本数、置信区间或方差，并做错误分类。

参考 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。

### 7.4 结构化输出、幻觉与安全边界

结构化输出的可靠性层级：提示约束 < JSON repair/重试 < schema 校验 < 受限解码/grammar。流程应是“模型生成 → parse → JSON Schema/Pydantic 校验 → 业务校验 → 可重试错误返回”，不能把能解析等同于内容正确。

幻觉至少分为：与给定证据冲突、无证据扩写、事实错误、引用错位。降低方法包括检索证据、要求可验证引用、允许 abstain、工具校验和任务专用评测；temperature 设为 0 只减少采样随机性，不保证事实正确。

输入/检索文档/工具返回都可能含 Prompt Injection。把外部文本视为不可信数据：系统规则与数据分隔、工具白名单、最小权限、参数校验、输出编码、敏感操作人工确认，并对“忽略之前指令”“读取密钥”等攻击建立回归集。

---

## 8. RAG：检索增强生成

### 8.1 数据流

```text
文档 → 解析/清洗 → chunk → embedding → 向量/关键词索引
问题 → query rewrite → 混合召回 → rerank → context packing → LLM → 引用答案
```

RAG 的价值是注入可更新、可追溯的外部知识，而不是保证绝不幻觉。每层都应单独评测。

### 8.2 Chunk、召回与重排

- 固定 token chunk 简单；按标题/段落/语义切分保留结构。
- overlap 能缓解边界信息丢失，但增加重复与索引成本。
- Dense retrieval 擅长语义；BM25 擅长精确词、编号、专名；混合检索通常更稳。
- Reranker 对候选做更精细的 query-document 相关性判断，以额外延迟换精度。
- Context packing 要去重、保留元数据、控制总 token，并避免不同来源互相冲突。

### 8.3 指标

- Recall@K：相关文档是否出现在前 K；依赖标注的 relevant set。
- MRR/nDCG：考虑相关文档排名；nDCG 可处理分级相关性。
- Faithfulness：回答陈述是否能由给定 context 支持。
- Answer correctness/relevance：回答是否正确、是否回答问题。
- Citation precision/recall：引用是否支持陈述、应引用的陈述是否有引用。
- 端到端还要记录 TTFT、P95 latency、token cost。

### 8.4 典型失败与排查

1. 没召回：检查解析、chunk、embedding、query rewrite 和 top-k。
2. 召回了但排序后丢失：检查 reranker 与过滤规则。
3. context 有答案但模型答错：检查 prompt、位置偏置、冲突内容和模型能力。
4. 答案正确但引用错：引用映射/生成后处理问题。
5. 更新后仍旧答案：索引增量、缓存和文档版本问题。

### 8.5 Embedding、索引与高级 RAG

**Embedding 训练与评测**：常用双塔对 query/document 编码，以 InfoNCE/多负样本对比损失拉近正例、推远负例：

$$L_i=-\log\frac{\exp(sim(q_i,d_i^+)/\tau)}{\sum_j\exp(sim(q_i,d_j)/\tau)}$$

in-batch negatives 便宜，但同批可能含假负例；hard negatives 应“相关但不回答”，可由 BM25/dense 召回后人工或规则过滤。评测先看 Recall@K/MRR/nDCG，再看端到端答案；不要只看通用 MTEB 排名。

**BM25 与 HNSW**：BM25 用词频饱和和文档长度归一化计算稀疏相关性，适合专名、编号、错误码。HNSW 是多层近邻图 ANN：`M` 影响图连接/内存，`efConstruction` 影响建库质量与耗时，`efSearch` 影响查询 recall/latency。它不是 embedding 模型。混合检索可归一化分数，或用 Reciprocal Rank Fusion：`RRF(d)=Σ_r 1/(k+rank_r(d))`。

**高级失败模式与处理**：

- Lost in the Middle：相关证据位于长 context 中间时使用率可能下降；提高 rerank、去噪，将关键证据靠前/靠后，缩短 context，并做位置交换实验。
- 多跳问题：先判断是否需分解，再对子问题检索、去重并合成；每一跳保存支持证据，避免错误级联。
- GraphRAG：抽取实体/关系/社区摘要，适合跨文档关系与全局问题；成本高、抽取误差会传播，不应替代所有向量检索。
- 增量更新：文档使用稳定 ID、内容 hash、版本、有效时间；先 upsert 新 chunk，再原子切换版本，最后清理旧索引/缓存。
- 时间衰减：相关性可加 `exp(-λΔt)` 或业务有效期，但政策/手册应按版本有效性而非简单“越新越好”。
- 表格/图片：保留页码、标题、行列关系、bbox 与 OCR 置信度；同时索引结构化表述和原始区域，回答引用回原页面。
- 噪声与冲突：低 OCR 置信、重复、过期、恶意指令应过滤/降权；冲突证据按权限、版本和发布日期处理，并显式告知不确定性。

决策原则：知识需要频繁更新/引用时优先 RAG；行为/格式迁移用 SFT；固定语料的检索失败不应靠“换更大生成模型”掩盖。

**自检**：为什么只看最终回答无法定位 RAG 问题？如何建立 100 条带证据的测试集？

---

## 9. Agent、工具调用与记忆

### 9.1 Agent 的最小闭环

Agent = 模型 + 状态 + 工具 + 控制循环 + 终止条件 + 可观测性。模型负责决策不等于所有流程都应交给模型；可确定的业务规则应写成代码或状态图。

```text
接收任务 → 规划/选择动作 → 校验工具参数 → 执行工具
        ↑                              ↓
  达到终止条件 ← 更新状态/观察结果 ← 处理错误
```

工具应有清晰 JSON schema、最小权限、超时/重试、幂等设计和返回大小限制。涉及写操作、付费、高风险动作时加入 human-in-the-loop。

### 9.2 ReAct、工作流与多 Agent

- ReAct：模型交替推理、行动、观察，适合开放任务，但必须限制步数与成本。
- Workflow：预定义图，稳定、可测；LangGraph 强项是状态化图和持久化。
- Multi-agent：按角色分工，但增加消息成本、错误传播和评测难度；单 Agent 能完成时不要为了“高级”而拆分。

框架定位：

| 框架 | 强项 | 适合 |
|---|---|---|
| LangChain | 模型、检索、工具的通用组件 | 快速搭建 LLM 应用 |
| LangGraph | 有状态、可恢复、长流程图 | 复杂 Agent/人工介入 |
| LlamaIndex | 数据摄取、索引、检索抽象 | 知识库/RAG |
| AutoGen | 对话式单/多 Agent 与事件运行时 | 多角色协作研究 |

接口变化快，使用时以 [LangGraph](https://github.com/langchain-ai/langgraph)、[LlamaIndex](https://docs.llamaindex.ai/) 和 [AutoGen](https://microsoft.github.io/autogen/stable/) 官方文档为准。

### 9.3 记忆

- 短期记忆：当前会话状态/消息；需裁剪或总结。
- 长期记忆：跨会话事实、偏好或历史事件；需要写入策略、检索和遗忘机制。
- 语义记忆保存事实，情景记忆保存事件，程序性记忆保存规则/技能。

风险：错误信息被永久写入、用户间数据串线、摘要丢关键约束、Prompt Injection 污染记忆。写入长期记忆应经过类型校验、来源记录、置信度和用户权限检查。

### 9.4 Agent 评测

记录 final success、每步轨迹、tool selection、argument accuracy、无效/重复调用、步数、token、成本、延迟和安全违规。用固定 sandbox 与 mock 工具保证可重复。失败按 planning、tool、observation、memory、termination 分类。

### 9.5 规划模式、12-Factor、MCP/A2A 与生产边界

| 方法 | 核心思路 | 适用与局限 |
|---|---|---|
| CoT | 生成中间步骤 | 简单但可能冗长/错误累积，不应要求暴露私有推理 |
| ToT | 搜索多个 thought 分支并评估 | 适合可评分规划，调用成本高 |
| Reflexion | 根据失败反馈形成反思再尝试 | 需要真实反馈；错误自评会强化偏差 |
| ReWOO | 先规划带变量依赖的工具步骤，再执行/合成 | 可减少交替调用，但环境变化时计划可能过时 |

“12-Factor Agents”是一套工程原则而非协议：自主管理 prompt/context/control flow；将工具调用视为结构化控制输出；统一业务状态与执行状态；支持 pause/resume 与 human-as-tool；把错误压缩进上下文；用小而专注的 agent；让 reducer 尽量无状态可重放。面试时应说明你采用了哪些原则及证据，不必机械背十二条。

**MCP** 是 host-client-server 的上下文/工具互操作协议：server 暴露 tools、resources、prompts 等能力，client 与 server 建立会话，消息基于 JSON-RPC。它解决发现与连接标准化，不自动提供业务授权、安全沙箱或幂等性。以当前正式 [MCP 2025-11-25 规范](https://modelcontextprotocol.io/specification/2025-11-25/basic) 为准；版本协商、能力声明和传输实现必须匹配。

**A2A** 面向独立 agent 之间的任务协作与状态/产物交换；MCP 更偏 agent/host 接入工具和上下文。两者可组合，不是替代关系。实现前固定 [A2A v0.3.0 规范](https://a2a-protocol.org/v0.3.0/specification/)；不要只凭同名 SDK 推断协议行为。

生产必做：

- 使用 JSON Schema/Pydantic 验证工具参数，错误返回短、结构化、可重试。
- context 超限前做基于任务的裁剪/摘要，保留系统约束、未完成事项、关键证据和工具结果引用；摘要应可追溯原消息。
- 工具运行在最小权限 sandbox；网络、文件、命令、凭证按工具隔离，写操作使用 idempotency key 与人工确认。
- 每次执行有 `tenant_id/session_id/run_id`，状态更新使用版本号/乐观锁；并发工具结果按 call id 合并，禁止共享可变全局记忆。

### 9.6 Agent 评测的可复现抽象

- **Task**：一个问题、初始环境与成功标准。
- **Trial**：某个 agent 配置在 task 上的一次独立运行。
- **Transcript**：消息、模型决策、工具调用、观察、时间与 token 的完整轨迹。
- **Outcome**：结束后的环境状态和最终产物。
- **Grader**：从 transcript/outcome 计算分数，可为代码规则、模型或人工。
- **Harness**：创建隔离环境、运行 trials、收集日志并聚合结果的系统。

Outcome grader 判断“结果是否完成”，Transcript grader 判断“过程是否合规/高效”；二者都要有。模型 grader 应与人工标注集做一致性、位置交换和阈值校准，不能自称客观。

若单次独立成功概率为 `p`，至少一次成功的 `pass@k≈1-(1-p)^k`；`pass^k≈p^k` 表示 k 次全部成功，衡量稳定性。真实估计使用每 task 多 trial 再宏平均，并报告置信区间；高温多采样抬高 pass@k 可能同时拉低 pass^k。用户模拟器适合测试澄清、多轮约束，但需固定 persona/隐藏目标，并用真实对话抽样验证其代表性。

稳定环境要求：容器/依赖版本、测试数据 snapshot、时间/随机种子、mock 外部服务、网络策略和清理逻辑固定；trial 间必须还原状态，防止前一次写入污染下一次。

---

## 10. 多模态大模型

### 10.1 基本结构

典型视觉语言模型：视觉编码器（如 ViT）把图像变为 patch tokens，经 projector/adapter 映射到 LLM embedding 空间，再与文本 token 一起输入语言模型。训练阶段可能包括：

1. 冻结两端，只训练 projector 做模态对齐。
2. 指令微调 projector + 部分/全部 LLM/vision encoder。
3. 偏好优化或面向 OCR、图表、视频等专项训练。

LLaVA、InternVL、Qwen-VL 的视觉 token 组织、动态分辨率、位置编码和训练配方不同，应以对应论文和版本模型卡为准。

### 10.2 数据处理

- 图像：resize/crop/normalize，注意长宽比、分辨率与 OCR 小字。
- 文本：问题、答案、图像占位符必须符合 chat template。
- 多图/视频：帧采样、时间顺序和 token 预算。
- 语音：声学特征或音频 encoder 输出与文本空间对齐。
- 动作：离散 action token 或连续控制头；需要环境反馈。

数据质量检查包括坏图、重复图、图文不匹配、答案不可见、OCR 泄漏、类别失衡。多模态 batch 常因图片尺寸不同产生负载不均。

### 10.3 训练策略与评测

冻结视觉 encoder 更省显存、稳定，但领域视觉差异大时上限受限；解冻后需更小学习率并防止视觉能力退化。可分别给 projector、LLM、vision tower 设置学习率。

评测不仅看自动分数：OCR、图表、空间关系、细粒度识别、幻觉都需分桶。避免模型利用文本偏差而不看图，可做遮图/换图反事实测试。

实践参考 [InternVL](https://github.com/OpenGVLab/InternVL)；它提供 Quick Start、Finetune、Evaluate、Deploy 路径。

### 10.4 CLIP/InfoNCE、融合、Grounding、视频与多模态 RAG

CLIP 用图像 encoder 与文本 encoder 产生归一化全局向量，对 batch 内图文相似度矩阵做双向 InfoNCE：每张图的配对文本、每段文本的配对图是正例，其他为负例。温度控制分布尖锐度。它学的是对齐的检索空间，不直接生成文本；假负例、batch 组成和数据噪声会影响训练。

融合范式：

- 双塔/late fusion：图文独立编码，适合大规模检索。
- projector + token 拼接：视觉 tokens 映射到 LLM，结构简单、生成常用。
- cross-attention/interleaved：语言层按需读取视觉特征，交互强但实现与计算更复杂。

Grounding 要把文字实体与图像区域/坐标关联。数据需保留 bbox/point/segmentation 与坐标归一化约定；指标可用 IoU、point accuracy、phrase grounding recall，并与回答正确率分开。只会描述图不等于会定位。

高分辨率常用 resize、tiling/dynamic resolution、缩略图+局部块或 token pruning。分块能保留小字但增加视觉 token，且必须保留全局布局和块坐标。视频还需帧采样、时间位置、运动特征与跨帧一致性；均匀采样可能漏掉短事件，密集采样又会耗尽 context。

多模态 RAG 可同时索引页面文本、表格结构、图片/区域 embedding 和元数据；query 先路由到文本或视觉召回，再 rerank、取原始页面区域给 VLM。评测包含跨模态 Recall@K、引用区域正确率、OCR/表格答案与视觉反事实。遮图、换图、打乱帧序能检查模型是否真的使用视觉输入。

---

## 11. 世界模型、强化学习与论文复现

### 11.1 世界模型

世界模型学习环境状态转移或观测分布，用于预测未来、规划或生成 imagined rollouts。关键概念：latent state、transition/dynamics model、observation model、policy/value。LLM/VLM 与世界模型结合时，语言 token 不必等同于真实物理状态；评测要关注可控性、一致性和长时误差累积。

### 11.2 RL 基础

MDP 包含状态 `s`、动作 `a`、转移 `P`、奖励 `r`、折扣 `γ`。价值函数衡量期望累计回报；policy gradient 使用：

$$\nabla_\theta J(\theta)=E[\nabla_\theta\log\pi_\theta(a|s)\hat A]$$

PPO 用 clipped surrogate 限制过大策略更新。LLM 中一个 token 可视为一步动作，但序列长、动作空间大、奖励稀疏，因此需要 KL、优势估计、reward normalization 和稳定 rollout 系统。

### 11.3 论文复现流程

1. 写一页 paper card：问题、贡献、假设、公式、数据、基线、指标。
2. 找官方代码，固定 commit；先跑最小样例和作者 checkpoint。
3. 对齐数据预处理、prompt/template、生成参数与评测脚本。
4. 先复现一个主结果，再做一个消融；报告差异而非隐藏失败。
5. 输出复现矩阵：原论文值、复现值、误差、硬件、随机种子、可能原因。

面试可说：“我先建立最小可证伪目标，再逐层对齐数据、模型、训练和评测；无法一致时通过中间统计定位，而不是直接调参碰运气。”

---

## 12. 8–12 周学习路线

| 周 | 学习主题 | 必做产出 |
|---|---|---|
| 1 | Python、PyTorch、数学/ML、Linux/Git、实验规范 | 完整训练循环、Softmax/CE 单测与环境报告 |
| 2 | Attention、MHA/MQA/GQA、FlashAttention、Tokenizer | 手写 Attention/online softmax；复杂度实验 |
| 3 | 预训练、数据、模型家族 | Mini Transformer 可生成文本 |
| 4 | SFT、LoRA/Adapter、QLoRA、蒸馏 | PEFT 参数/显存/效果对照 |
| 5 | DPO、RM、PPO/GRPO、评测 | SFT/DPO 对照；GRPO paper card 或选修实验 |
| 6 | DDP、AMP、ZeRO/FSDP、collective | 1/2/4 卡 scaling 与通信报告 |
| 7 | Embedding、hybrid/多跳/GraphRAG | 带检索分层指标与更新/注入测试的知识库 |
| 8 | Agent、MCP、状态/安全/评测 | 有恢复、隔离、grader 和多 trial 的 Agent |
| 9 | CLIP、Grounding、高分辨率、多模态 RAG | VLM 推理/LoRA 与视觉反事实 |
| 10 | 论文复现、综合评测 | 复现报告与消融 |
| 11 | 面试题与项目表达 | 两次录音模拟面试 |
| 12 | 补弱项、整理 GitHub/简历 | 可复现 README、Demo、真实指标 |

若只有 8 周：合并 3/4、5/6、9/10，并保留项目的评测和复盘，不要只跑 Demo。

---

## 13. 速查附录

### 13.1 常见显存优化优先级

1. 减小 micro-batch/sequence length，开启 gradient checkpointing。
2. BF16/FP16、FlashAttention/SDPA、packing。
3. LoRA/QLoRA，8-bit optimizer。
4. ZeRO/FSDP，必要时 CPU offload。
5. 重新选择模型规模；不要用极端 offload 把不可行训练变成数周慢任务。

### 13.2 高频术语

| 术语 | 一句话解释 |
|---|---|
| SFT | 用理想示范做监督式后训练 |
| PEFT | 只训练少量附加/选定参数的微调方法族 |
| LoRA | 用低秩矩阵表示权重增量 |
| QLoRA | 4-bit 冻结基座 + 可训练 LoRA |
| DPO | 直接用成对偏好优化策略，无需显式 RM+PPO 链路 |
| RAG | 检索外部证据后再生成 |
| GQA | 多个 Query heads 共享较少 KV heads |
| ZeRO | 分片训练状态降低每卡冗余显存 |
| TP/PP/DP | 张量/流水线/数据并行 |
| TTFT/TPOT | 首 token 延迟/每输出 token 延迟 |
| Agentic RAG | Agent 决定何时、如何、多轮检索 |

### 13.3 最终能力检查

- [ ] 能在白板写出 Attention、LoRA、DPO 的关键公式并解释每项。
- [ ] 能从零写出带 AMP、累积、裁剪、验证和 checkpoint 的训练循环。
- [ ] 能解释一次 OOM、NaN、低吞吐的真实排查过程。
- [ ] 能建立检索层和生成层分开的 RAG 评测。
- [ ] 能展示 Agent 的状态、轨迹、终止与失败恢复。
- [ ] 能说明 VLM 中视觉 token 如何进入 LLM，以及冻结策略的权衡。
- [ ] 能用日志、配置、指标和错误样本证明项目结论。

### 13.4 本地面试资料审计与纠错原则

本轮扩充分析了 `interview/` 中三套 GitHub 资料（共 306 篇 Markdown、7 个 Notebook）以及一份 12,960 条自动归并题目集。它们用于发现考点、追问方式和实践线索；技术结论仍回到论文、规范和官方文档核验。访问日期：2026-07-14。

| 资料库 | 对本指南的主要价值 | 局限/处理 | 许可与引用策略 |
|---|---|---|---|
| AgentGuide | Agent/RAG 工程、评测、安全、近年面经覆盖广 | 部分章节是经验总结，前沿名词变化快 | 根目录未发现明确许可证；仅归纳主题，不复制表达 |
| FAQ_Of_LLM_Interview | 真实面经、Transformer/微调/RAG 高频追问 | 匿名记录、答案质量不一，存在架构绝对化与来源错误 | MIT；改写并用 primary source 纠错 |
| LLMForEverybody | 推理优化、量化、RAG、MCP 与 Notebook 实验思路 | 旧 Notebook 含过时 LangChain/OpenAI API 和模型名 | Apache-2.0；只迁移概念与实验设计 |
| 自动归并题目集 | 12,960 条题目便于发现重复主题 | 聚类/频次受同义题、转载与噪声影响，不代表招聘概率 | 仅作定性优先级，不发布公司频次结论 |

明确纠正：RoPE 来源应引用 RoFormer，而非本地材料中的错误名称；模型结构必须限定到具体版本；DPO/GRPO 参数必须结合目标函数、采样和 reward 讨论；旧 `initialize_agent`、旧 OpenAI SDK、硬编码密钥写法不进入示例。所有“更快/更好”结论必须写硬件、数据、版本、基线与指标。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT](https://arxiv.org/abs/1810.04805)、[LLaMA](https://arxiv.org/abs/2302.13971)
- [LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)
- [DPO](https://arxiv.org/abs/2305.18290)、[InstructGPT/RLHF](https://arxiv.org/abs/2203.02155)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)与[官方仓库](https://github.com/QwenLM/Qwen3)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)
- [vLLM](https://docs.vllm.ai/)、[MCP 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic)、[A2A v0.3.0](https://a2a-protocol.org/v0.3.0/specification/)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [DeepSpeed](https://www.deepspeed.ai/)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [LangGraph](https://github.com/langchain-ai/langgraph)、[InternVL](https://github.com/OpenGVLab/InternVL)

> 访问日期：2026-07-14。框架 API 更新快，运行前应再次检查官方安装与迁移说明。
