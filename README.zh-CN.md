# Policy Split

**通过双模式熵正则化激励 LLM 强化学习中的双模式探索**

[📄 论文](https://arxiv.org/abs/2606.04701) · [💻 代码](https://github.com/BITHLP/PolicySplit) · [English](README.md) | **简体中文**

---

面向 LLM 的熵引导强化学习存在一个根本性的探索–利用冲突：一味追求更高的熵往往会损害准确率，在对本就熟练的低熵大推理模型（LRM）进行持续后训练时尤为明显。**Policy Split** 通过把单个模型分叉为两个协作的策略来化解这一冲突——一个面向正确率优化的**常规模式**，与一个偏好探索的**高熵模式**——二者通过**双模式熵正则化**协同训练。

<p align="center"><img src="assets/intro.png" width="62%"></p>
<p align="center"><em>(a) 朴素地偏好高熵会导致准确率下降。(b) Policy Split 将策略分叉为常规模式与高熵模式（由高熵系统提示标识），二者共享参数与 rollout。(c) 双模式熵正则化使两种模式表现出不同的熵行为，并产生持续增大的 KL 散度。</em></p>

## 核心思想

在一个模型内部实例化两个互相借鉴经验的策略：

- **常规模式** `π_θ(·|q)`——使用原始 GRPO 优势，**只面向正确率**训练。
- **高熵模式** `π_θ^HE(·|q) = π_θ(·|s,q)`——共享同一套参数，由高熵系统提示 `s` 激活，使用**带熵正则的优势**训练，该优势额外加入 (i) 熵偏好项与 (ii) 推动其远离常规模式的 KL 项，并通过 clamping 保证稳定性。
- **策略间协作**通过 **rollout 共享**实现：常规模式可学习新发现的正确 rollout，而高熵模式借助稳定的正确回答避免坍缩——代价是约 1.4× 于 GRPO 的训练时间。

> 注意：仅仅在上下文前拼接一个高熵提示是**不够**的——输出分布几乎不变（KL ≈ 0.01）。必须经过显式训练才能建立起真正的高熵模式。

<p align="center"><img src="assets/method.png" width="100%"></p>
<p align="center"><em>Policy Split 训练框架：从两种模式分别采样 rollout 并共享，再各自分配对应的「仅正确率」或「熵正则化」优势。</em></p>

## 核心亮点

- **稳定的提升**：在 Qwen3-1.7B/4B/8B 上的平均准确率均超过强熵引导 RL 基线（相对基座模型的所有提升均通过 *p* < 0.01 的显著性检验），同时**复活了**低熵 LRM 的熵。
- **一模两用**：常规模式更严谨、擅长通用任务；高熵模式擅长创意写作——推理时只需切换提示即可。
- **真正的双模式分化**：Policy Split 大幅拉开两模式间的 KL 散度与行为差距，而对原始模型或朴素 GRPO 加提示几乎不改变行为。
- **更高的 Best-of-N**：高熵模式能发现更多**独特的**正确 rollout（best-of-8 更高），为训练提供差异化的学习信号。
- **一定的提示泛化能力**：改写后的高熵提示仍能诱导出高熵，但遵循未见过的低熵指令仍是瓶颈。

## 方法速览

| 组成 | 常规模式 | 高熵模式 |
|------|---------|---------|
| 激活方式 | 无系统提示 | 高熵系统提示 |
| 优势计算 | 原始 GRPO（仅正确率） | GRPO + clamped 熵项与 KL 项 |
| 优化目标 | 保持准确率 | 鼓励新颖探索 |
| 最佳场景 | 通用 / 精确任务 | 创意任务 |

两种模式共享参数与 rollout，且无论某条 rollout 由哪种模式产生，始终按其所属模式的优势类型训练。

## 引用

```bibtex
@article{yao2026policy,
  title={Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization},
  author={Yao, Jiashu and Huang, Heyan and Wu, Daiqing and Liu, Zeming and Guo, Yuhang},
  journal={arXiv preprint arXiv:2604.11510},
  year={2026}
}
```
