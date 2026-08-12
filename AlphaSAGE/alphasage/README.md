# AlphaSAGE · 学习包

> **AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration**
> Binqi Chen, Hongjun Ding, Ning Shen, Taian Guo, Jinsheng Huang, Luchen Liu, Ming Zhang
> 北京大学 / 正仁量化 / Baruch College CUNY / UBC · ICLR 2026 · arXiv:2509.25055v3
> 代码：https://github.com/BerkinChen/AlphaSAGE

---

## 这篇论文讲什么

自动化 alpha 挖掘的主流路线是把"构造公式"建模成序列决策问题，用 RL 去搜。这篇论文指出这条路有三个**相互纠缠**的结构性缺陷：奖励只在公式建完时才有（稀疏）、把公式当 token 序列处理（丢掉结构）、以及最关键的——**RL 的 `max E[R]` 目标天然收敛到单一最优解，而量化研究需要的是一批彼此低相关的因子**。

最后这一条是全文的灵魂：**目标函数与实际需求存在结构性错配**。生产系统维护的是因子库，组合的价值来自因子之间的低相关性。「找一个最好的」和「找一批各不相同的」是两个不同的问题，而 RL 只能做前者。

于是作者换了引擎：用 **GFlowNet** 学 `P(α) ∝ R(α)`（按奖励比例采样，而非最大化），配上 **AST + RGCN** 的结构感知编码器和 **IC + 结构对齐 + 新颖性**的三项退火奖励。在 CSI300/500/1000 和 S&P500 上，组合指标全面领先——CSI300 的 Sharpe 从最强基线的 0.88 提到 1.71。

**一句话**：把 `argmax` 换成 `∝`，多样性就从外挂的正则项变成了目标函数的内在属性。

---

## 难度与用时

| | |
|---|---|
| **难度** | Advanced（需要 RL + 图神经网络 + 生成模型三方面基础） |
| **性质** | 新训练范式 + 架构设计 |
| **预计学习时间** | 精读 3~4 小时；跑完 demo 再加 1 小时 |
| **主要门槛** | **GFlowNet**。论文只用一段话介绍，但它是全文引擎 |
| **不需要** | LLM 相关知识 |

---

## 怎么用这个学习包

1. **[summary.md](summary.md)** — 三个诊断、三个药方、全部实验数字。*(25 分钟)*
2. **[index.html](index.html)** — 交互页。第一个标签直接演示模式坍缩，**比读公式快得多**。*(15 分钟)*
3. **[code/gflownet_diversity_demo.py](code/gflownet_diversity_demo.py)** — 亲眼看到 RL 坍缩到 1/4 模式而 GFlowNet 覆盖 4/4。*(10 分钟)*
4. **[method.md](method.md)** — 逐组件拆解 + 完整算法 + 复现陷阱。*(45 分钟)*
5. **[insights.md](insights.md)** — 批判性分析，包括几处论文说过头的地方。*(35 分钟)*
6. **[code/alpha_ast_demo.py](code/alpha_ast_demo.py)** — 结构编码的边类型设计 + 奖励冲突检验。*(15 分钟)*
7. **[qa.md](qa.md)** — 15 道自测题。*(35 分钟)*
8. **[mental-model.md](mental-model.md)** — 知识地图定位。*(15 分钟)*

**只有 30 分钟？** 读 insights.md 第 2 节（目标函数错配），然后跑一遍 `gflownet_diversity_demo.py`。这两样就是论文的核心。

---

## 核心要点（先记住这五条）

1. **诊断比药方更值钱**：「标准 RL 的目标函数与"需要一组多样解"的需求存在结构性错配」——这个观察适用于分子设计、NAS、程序合成等一切同类问题。

2. **最大的单项提升来自 GNN**（IC 0.046 → 0.070，SR −0.11 → 1.25）。论文的叙事重心在 GFlowNet 上，但消融告诉你：**如果只能改一件事，改编码器**。

3. **`R_NOV = 1 − max|IC(α, α')|` 是最容易单独抄走的组件**，不依赖 GFlowNet，可直接加进任何现有搜索框架。

4. **两个市场的回测规则不同**（CSI 仅做多 top 20%，S&P500 是 top/bottom 10% 多空），所以 **SR 不可跨市场比较**——S&P500 的 6.32 是对冲的结果，它的 IC 反而更低。

5. **ES 和 GNN 是耦合的**：单独启用 early stopping 会让 AR 从 3.63% 掉到 **−0.47%**（SR 变负）。

---

## 论文的三个缺口（本包做了实测）

- **核心主张从未被直接验证**。因果链是「GFlowNet → 因子更多样 → 组合更好」，起点和终点都验证了，**中间那一环全文没有一个数字**。存在说得通的替代解释：GFlowNet 可能只是个更好的优化器。补这个实验成本极低（见 [qa.md 第 12 题](qa.md)）。

- **「AST 对语法变体不变」这句话不完整**。实测（`alpha_ast_demo.py`）显示：不变性其实来自 GNN 邻居聚合的置换不变性，而论文只提了 `data`/`window` 一个边类型维度——**光靠它无法区分 `Sub(a,b)` 与 `Sub(b,a)`**，会把两个不同的公式判成相同。复现时必须自己把交换律处理对。

- **`R_SA` 与 `R_NOV` 存在结构性负耦合**。论文 §3.4 专门论证三项奖励"没有显式的负耦合"，但 `R_SA` 的 K 近邻正是从因子库里找的——行为上真正新颖的 alpha，其嵌入近邻在行为上必然离它很远，`R_SA` 因而被压低。实测 `corr(R_SA, R_NOV) = −0.473`。这可能正是论文自己观察到的「新颖性权重过大时性能下滑」的原因。

  > 保留意见：实测在合成数据上做，是定性演示不是证伪。但方向真实，验证成本只有几行代码。

**另外几处措辞超出数据支持**：「所有相关性指标第一」有至少两处反例（CSI500 的 ICIR、CSI1000 的 RICIR）；熵正则让 RICIR 从 0.614 掉到 0.583，论文只说"整体最好"没讨论。

---

## 目录结构

```
alphasage/
├── README.md              ← 你在这里
├── summary.md             三个诊断 / 三个贡献 / 全部实验数字
├── insights.md            核心洞察、权衡、9 条局限性
├── method.md              逐组件拆解 + 完整算法 + 10 条复现陷阱
├── mental-model.md        知识地图 + 两个层面的"多样性"辨析
├── qa.md                  15 道自测题（含详细解答）
├── index.html             交互式探索器（4 个标签页，零依赖）
├── meta.json              元数据
├── paper.pdf              原文
├── code/
│   ├── gflownet_diversity_demo.py  RL vs GFlowNet 的模式坍缩对比
│   └── alpha_ast_demo.py           AST 边类型设计 + 三项奖励计算
└── images/
    ├── architecture.jpeg           Figure 2 框架总览
    ├── cumulative_return_csi300.jpeg  Figure 3 累计收益
    └── sensitivity_rewards.jpeg    Figure 4 奖励权重敏感性
```

**代码说明**：本机装不上 numpy/torch（沙箱网络受限），两个 demo 都是**纯标准库、可直接 `python3` 运行**。GFlowNet demo 用 tabular softmax 手推梯度，两种方法共用同一套参数化，**唯一的变量是目标函数**——这正是论文想说的事。实测结果：RL 的 KL(目标‖学到) = 3.30 且只发现 1/4 个模式，GFlowNet 的 KL = 0.0000 且发现 4/4。

---

## 同批学习的另外三篇

| | 路线 | 一句话 |
|---|---|---|
| [StockMixer](../stockmixer/) | A：端到端黑箱 | 用带约束的 MLP 替代 RNN+GNN 混合架构 |
| [VTA](../vta-technical-analysis/) | A：端到端 + 可解释 | LLM 生成技术分析推理链，再条件化时序骨干 |
| **AlphaSAGE** | B：生成公式因子 | GFlowNet + RGCN，追求因子多样性而非单一最优 |
| [AlphaBench](../alphabench/) | B：评测 | 系统测量 LLM 在公式化因子挖掘上到底行不行 |

**最该对照读的是 AlphaBench**：它发现 LLM 极不擅长"不做回测就判断因子好坏"（准确率接近随机猜），这反过来支持了 AlphaSAGE 的路线——与其让模型凭直觉判断，不如把 IC 直接做成奖励信号。

详见 [mental-model.md §6](mental-model.md)。
