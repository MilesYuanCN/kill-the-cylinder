> 因为阶段二里面， 包含了一些阶段三的内容， 并且题干内容没那么多， 所以让其将阶段三和四合并到一起了
> 
> ### **合并阶段：策略梯度→PPO核心逻辑（关键路径）**
> 
> #### **Step 1：策略梯度核心问题（1小时）**
> 
> *   **核心目标**：理解策略优化中的两个关键问题
>     1.  **高方差问题**：策略梯度依赖轨迹采样，方差大→需要降低方差
>     2.  **更新步长问题**：策略更新过大易崩溃，过小收敛慢→需要约束更新
> *   **必须掌握**：
>     *   **Advantage Function**：用 A(s,a)=Q(s,a)−V(s)A(s,a)\=Q(s,a)−V(s) 替代原始回报，降低方差
>     *   **重要性采样**：用旧策略的数据评估新策略（PPO的关键基础）
> *   **跳过内容**：REINFORCE算法细节、蒙特卡洛方法推导
> 
> #### **Step 2：TRPO到PPO的进化（1小时）**
> 
> *   **TRPO的核心思想**（20分钟）：
>     *   用**信任域**约束策略更新：强制新策略与旧策略的KL散度小于阈值
>     *   问题：二阶优化计算复杂，实现困难
> *   **PPO的解决方案**（40分钟）：
>     *   用\*\*概率比裁剪(Clip)\*\*替代KL约束（核心创新）
>     *   目标函数设计：
>         
>         LCLIP=E\[min⁡(πθ(a∣s)πθold(a∣s)A(s,a),clip(πθπθold,1−ϵ,1+ϵ)A(s,a))\]LCLIP\=E\[min(πθold​​(a∣s)πθ​(a∣s)​A(s,a),clip(πθold​​πθ​​,1−ϵ,1+ϵ)A(s,a))\]
>         
>     *   **直观解释**：限制新策略与旧策略的动作概率比，防止突变
> *   **对比动画理解**（重点！）：
>     *   未裁剪的更新：策略可能大幅偏离导致崩溃  
>         ![PPO Clip机制](./跟着 DS 学习 RL（三）PPO & GRPO - 原禄城 - Confluence_files/ppo-clip.png)
> 
> #### **Step 3：PPO实现关键（1-2小时）**
> 
> *   **三大组件**：
>     1.  **Clipping机制**：ε通常取0.1-0.3（如DeepSeak可能用0.2）
>     2.  **Advantage归一化**：对Advantage值做标准化（减均值除标准差）
>     3.  **多轮经验复用**：用同一批数据更新策略多次（通常3-10次）
> *   **代码逻辑伪代码**（理解即可）：
>     
>         for _ in range(update_epochs):
>             advantages = (returns - values).normalize()
>             ratio = new_probs / old_probs
>             clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
>             loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
>             optimizer.step()
>     

TRPO
====

2017/4/20 加利福尼亚大学

Trust Region Policy Optimization: 信任地区优化

资料中，只给了个论文。

*   [https://arxiv.org/abs/1502.05477](https://arxiv.org/abs/1502.05477)，说是让读其 introducing 就行。
*   [openai TRPO doc](https://spinningup.openai.com/en/latest/algorithms/trpo.html)

  

Some facts：

*   TRPO 采取了在限制内的最大步幅来更新策略， 这个限制是 KL-Divergence（什么相对熵）：算不动概率分布之间的距离
*   TRPO 与普通的 policy gradient 不一样， 它使得两次迭代的参数空间非常相近。
    *   但是即使很小的参数空间差异也可能会导致性能的大差异。（ VPG 就会导致 clapse（不在一个碗里收敛））
    *   这样也就会避免这种崩溃
*   TRPO 是一种 on-policy 同策略算法
*   TRPO 可用于连续和离散的动作空间
*   TRPO 的旋转支持使用 MPI 来并行计算（完全不知道是啥）

此处数学计算就开始复杂起来了，以下是一些我需要重新复习的名词....

*   Hessian 矩阵
*   泰勒展开
*   拉格朗日二元性
*   共轭梯度

感觉这里学习梯度有点高， 在这里就不求甚解了。总之， TRPO 解决的问题就是：

*   step 太大， 导致一步迈出去了， 一直收敛不了

带来的问题是：

*   计算复杂度较高， 对算力要求高。

  

PPO：
====

*   论文：[PPO原始论文（重点读Section 3）](https://arxiv.org/abs/1707.06347)
*   代码：[PPO伪代码解读](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
*   [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

2017/8/28 由 openai 提出

动机和 TRPO 一样， 都是防止步子大了迈出去了收敛不了。

TRPO 用了复杂的二阶方法， PPO 用了一堆一阶的 Trick 来搞。 PPO 的实现简单， 并且实践上来说和 TRPO 效果一样。

有两种变体：

*   PPO-PENALTY
    *   求解 KL-constrained 类似于 TRPO， 不过不是严格小于， 只是增加了惩罚系数。
*   PPO-CLIP
    *   没有任何 KL-constrained 东西，而是在 「objective function」进行裁剪， 来减少步子太大的可能性（没懂）

Some Facts：

*   PPO 是一种同策略算法。
*   可以用于离散或者连续的动作空间
*   也支持 MPI 的并行化

![](./跟着 DS 学习 RL（三）PPO & GRPO - 原禄城 - Confluence_files/image-2025-2-10_21-23-11.png)

![](./跟着 DS 学习 RL（三）PPO & GRPO - 原禄城 - Confluence_files/image-2025-2-10_21-22-2.png)

大概能懂这个方程含义： 

*   目的是找到最大化 L 期望时的参数
*   L 的含义是， 优势函数加个 clip
    *   如果优势 > 0， 那它不能超过  1 + 3（忘了读啥了）
    *   如果优势 < 0， 那它不能超过  1 - 3
*   所以， 感觉就是限制更新率。

GRPO
====

[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

DS 的知识库里面还没有 GRPO 相关支持（2024/2）， 可能只能读论文了。

本论文贡献：

数学的预训练模型：

*   爬了 120B 数据
*   7B 的 DeepSeekMath-Base 和 540B 的 Minerva 性能相当， 证明参数量不是能力的唯一主要因素。
*   先 Code Training 之后再 Math Training， 效果不错。 猜测是 code training 增加了其推理能力。
*   用 arXiv 论文好像并没有什么帮助

Exploration and Analysis of Reinforcement Learning：

*   介绍了 GRPO: Group Relative Policy Optimization
    *   其放弃了评论家模型， 从 goup scores 得到了 baseline， 相对于 PPO 大大减少了计算资源。
    *   在 DeepSeekmath 里面证明好使
    *    RFT DPO PPO GRPPO 进行了对比。
    *   探讨了几种更有效的 reinforcement 大模型的方式。

![](./跟着 DS 学习 RL（三）PPO & GRPO - 原禄城 - Confluence_files/image-2025-2-10_21-55-41.png)

主要为了解决两个问题：

*   价值模型（评论家）和策略模型基本上是同样的 Size，增加了记忆和计算负担。
*   在 LLM 上下文， 每个 token 来去优化价值模型是复杂的。

使用多个采样输出的平均奖励（基同一个问题）作为 baseline。 

![](./跟着 DS 学习 RL（三）PPO & GRPO - 原禄城 - Confluence_files/image-2025-2-10_22-0-17.png)

（没有读懂公式，  但大概知道在干什么）

有一些问题：

*   Critic-Actor 是为了解决方差大， 导致期望预测不准的问题， 以及将蒙特卡洛变为时间差异的问题。
*   这样去掉了评论家， 不会导致方差还是很大吗？  
    *   这里也是使用一堆平均值， 来去当做 baseline
    *   然后用 baseline 就能判断出来 Q 了（看起来是比例）
    *   （可能这种比较适用于 LLM？）