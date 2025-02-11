> ### **阶段2：经典算法分类（3-4小时）**
> 
> *   **目标**：区分基于价值、基于策略的算法
> *   **关键内容**：
>     1.  **基于价值的方法**：Q-Learning → DQN（经验回放、目标网络）
>     2.  **基于策略的方法**：策略梯度（Policy Gradient）直接优化策略
>     3.  **Actor-Critic框架**：结合价值函数和策略的混合方法
> *   **重点对比**：DQN vs 策略梯度 vs Actor-Critic
> *   **学习资源**：
>     *   视频：[Policy Gradient直观解释](https://www.youtube.com/watch?v=KHZVXao4qXs)
>     *   动手实践：[CartPole环境代码示例](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
> 
> * * *

这一部分给的过于简略， 又让其提供了份更加详细的大纲：

> 以下是专为你设计的 **3-4小时强化学习经典算法分类** 详细学习大纲，包含分层递进的学习步骤和精选资源：
> 
> * * *
> 
> ### **阶段2：经典算法分类（3-4小时）**
> 
> #### **目标**：掌握基于价值/策略/混合方法的区别，建立算法分类框架
> 
> * * *
> 
> ### **一、基于价值的方法（Value-Based） (1-1.5小时)**
> 
> #### **1\. Q-Learning核心原理**
> 
> *   **核心公式**：  
>     ( Q(s\_t,a\_t) \\leftarrow Q(s\_t,a\_t) + \\alpha \[r\_{t+1} + \\gamma \\max\_a Q(s\_{t+1},a) - Q(s\_t,a\_t)\] )
> *   **关键特性**：
>     *   表格法（Tabular）：适用于离散小状态空间
>     *   无模型（Model-Free）：无需环境动力学
> *   **动态演示**：  
>     [Q-Learning迷宫寻宝交互演示](https://www.ra.cs.uni-tuebingen.de/mitarbeiter/milicak/Q-learning/)
> 
> #### **2\. DQN的革新**
> 
> *   **三大改进**：
>     1.  **经验回放（Experience Replay）**：打破数据相关性
>     2.  **目标网络（Target Network）**：稳定训练
>     3.  **神经网络拟合Q函数**：处理高维状态
> *   **架构图**：  
>     ![DQN Architecture](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/1_HMu8J2zHG-C6oVjYII-Eqg.png)
> *   **关键论文**：  
>     [Nature DQN论文图表解读](https://www.nature.com/articles/nature14236/figures/1)（重点看图1）
> 
> #### **3\. 代码实践（20分钟）**
> 
> *   **CartPole环境实现**：
>     
>         import gym
>         from keras.models import Sequential
>         from keras.layers import Dense
>         
>         env = gym.make('CartPole-v1')
>         model = Sequential([
>             Dense(24, activation='relu', input_dim=4),
>             Dense(24, activation='relu'),
>             Dense(2, activation='linear')  # 输出Q值
>         ])
>         # 完整实现参考：https://github.com/keon/deep-q-learning
>         
>     
> 
> * * *
> 
> ### **二、基于策略的方法（Policy-Based） (1-1.5小时)**
> 
> #### **1\. 策略梯度定理**
> 
> *   **目标函数**：  
>     ( J(\\theta) = \\mathbb{E}_{\\pi_\\theta}\[\\sum \\gamma^t r\_t\] )
> *   **梯度公式**：  
>     ( \\nabla\_\\theta J(\\theta) = \\mathbb{E}\[\\sum \\nabla\_\\theta \\log \\pi\_\\theta(a|s) Q^\\pi(s,a)\] )
> *   **直观解释**：  
>     [Policy Gradient的梯度上升可视化](https://imgur.com/a/PolicyGradientGIF)
> 
> #### **2\. REINFORCE算法**
> 
> *   **蒙特卡洛特性**：
>     *   需完成完整episode后才能更新
>     *   高方差问题（对比DQN）
> *   **伪代码流程**：
>     
>         1. 用当前策略采集轨迹
>         2. 计算每个时间步的回报G_t
>         3. 梯度更新：θ ← θ + α ∑ G_t ∇logπ(a|s)
>         
>     
> 
> #### **3\. 对比实验**
> 
> *   **与DQN的差异**：
>     
>     特性
>     
>     DQN
>     
>     Policy Gradient
>     
>     输出维度
>     
>     每个动作的Q值
>     
>     动作概率分布
>     
>     探索方式
>     
>     ε-greedy
>     
>     策略的随机性
>     
>     连续动作支持
>     
>     需离散化
>     
>     直接支持
>     
> 
> * * *
> 
> ### **三、Actor-Critic框架 (1小时)**
> 
> #### **1\. 两模块架构**
> 
> *   **Actor**：策略网络 ( \\pi\_\\theta(a|s) ) → 生成动作
> *   **Critic**：价值网络 ( V\_\\phi(s) ) 或 ( Q\_\\phi(s,a) ) → 评估动作
> *   **结构示意图**：  
>     ![Actor-Critic](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/3707a5c06fe28b0c2660e70c46cce5bfd84dac90.svg)
> 
> #### **2\. 优势函数（Advantage）**
> 
> *   **定义**：  
>     ( A(s,a) = Q(s,a) - V(s) )  
>     （衡量特定动作相对于平均水平的优势）
> *   **计算技巧**：  
>     TD误差 ( \\delta = r + \\gamma V(s') - V(s) ) 作为优势估计
> 
> #### **3\. 经典变体**
> 
> *   **A2C（Advantage Actor-Critic）**：同步多环境采样
> *   **A3C（Asynchronous）**：异步多线程（已较少使用）
> *   **代码实践**：  
>     [PyTorch Actor-Critic实现](https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py)
> 
> * * *
> 
> ### **四、对比与选择指南（30分钟）**
> 
> #### **决策树：如何选择算法？**
> 
>     graph TD
>       A[问题特性] --> B{动作空间}
>       B -->|离散| C[DQN系列]
>       B -->|连续| D[Policy Gradient/PPO]
>       A --> E{是否需要分离策略评估}
>       E -->|是| F[Actor-Critic]
>       E -->|否| G[纯策略梯度]
>     
> 
> #### **性能对比表**
> 
> 指标
> 
> DQN
> 
> Policy Gradient
> 
> Actor-Critic
> 
> 样本效率
> 
> 中
> 
> 低
> 
> 高
> 
> 训练稳定性
> 
> 需目标网络
> 
> 高方差
> 
> 中等
> 
> 连续动作支持
> 
> 不支持
> 
> 支持
> 
> 支持
> 
> 并行化能力
> 
> 难
> 
> 中等
> 
> 易（A3C）
> 
> * * *
> 
> ### **精选学习资源**
> 
> 1.  **视频课程**：
>     *   [David Silver RL Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)（重点看前30分钟）
> 2.  **交互教程**：
>     *   [Deep Reinforcement Learning Demystified](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)（Section 3对比算法）
> 3.  **代码库**：
>     *   对比实现：[DQN vs PG vs AC in CartPole](https://github.com/dennybritz/reinforcement-learning)
> 
> * * *
> 
> ### **学习效果验证**
> 
> 完成学习后尝试回答：
> 
> 1.  为什么DQN需要目标网络而Policy Gradient不需要？
> 2.  Actor-Critic中的Critic是否可以完全替代价值函数？
> 3.  在机械臂控制任务中应该优先选择哪类算法？为什么？
> 
> 遇到卡顿时可重点复习算法对比表和架构图。建议边看代码边理解理论，实践可大幅提升理解速度

一些链接仍不可用， 但是在相关内容中， 已经找到了一些对应的内容。

[openai spinning up](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

[huggingface course](https://huggingface.co/learn/deep-rl-course/unit2/introduction)

先跟着 Openai 文档看下。

RL 算法种类分类
=========

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_12-42-37.png)

现代 RL 算法不是能用树结构很好分类的。 但是为了便于理解， 减少了一些内容。

以下是一些关键的「branching point」分支节点（分类维度）

Model-Free vs Model-Based RL
----------------------------

主要区别是， Agent 是否可以访问（或者学习）一个模型的环境。

拥有模型的好处是， Agent 可以提前思考， 预测行动后会发生什么以明智选择。

缺点是， 由于 Agent 一眼看不到模型真实的客观规律， 很可能欠拟合或过拟合。

模型的学习是客观困难的， 即使耗费很多的时间和计算资源有时也难以搞定。

Model-Based RL：

*   使用模型的 RL
*   效率潜在提高（有更多信息）

Model-Free RL：

*   放弃了使用模型
*   更易于实现和调整
*   减少了过拟合和欠拟合

  

What to Learn 要学习什么
-------------------

*   策略：随机的还是固定的
*   action-value functions：动作价值函数
*   value functions：价值函数
*   （可选）环境模型

  

### 对于 Model-Free RL 来说， 有两种方法：

*   Policy Optimization：
    *   策略优化， 优化策略中的参数 /theta。  
        ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_18-54-21.png)
    *   直接通过价值函数来梯度下降，或者简介的最大化局部价值函数 J()。
    *   因而， 这种优化基本上都是 on-policy （同策略）来执行。
    *   一些例子：
        *   A2C/A3C, 可以执行梯度上升来提高性能
        *   PPO，通过最大化 surrogate objective function， 来保守的估计价值函数如何变化来更新
*   Value function Optimizing ：Q-Learning & DQN：
    *   通过最大化动作价值函数， 来去找到最优策略。 一般来说这种优化总是和策略无关的。 即找到非同策略情况下的最优 
    *   了解哪种状态更有价值， 并在使用此值函数来采取导致该状态的动作。
    *   方法：
        *   DQN：后面会讲到
        *   C51

  

实际上， Policy Optimization 和 Q-Learning 并不矛盾， 有一些算法在两者之间。 并在两者的算法特性中进行 trade-off

*   DDPG：使用两种方法来优化另一种方法
*   SAC：使用一些其他策略， 最终表现好像比 DDPG 好一些。

### 对于 Model-Based RL 来说

不像 Model-free 的 RL，有比较确定的分类。 Model-Based RL 里面有很多正交的使用模型的方法。

*   Background: Pure Planning
*   Expert Iteration
*   Data Augmentation for Model-Free Methods
*   Embedding Planning Loops into Policies

  

（想偷懒， 不学 Q-Learning 和 Actor-Critic， 但 DS せんせ 说不学不行）

> ### **是否有必要学习Q-Learning和Actor-Critic框架？**
> 
> #### **1\. 是否需要学习Q-Learning？**
> 
> *   **必要性**：**是**，即使你主攻策略方法。
> *   **原因**：
>     *   **基础理解**：Q-Learning是强化学习的基石，理解其贝尔曼方程、时序差分（TD）更新等概念，是后续学习的基础。
>     *   **算法对比**：基于价值的方法（如DQN）与策略方法（如PPO）在探索方式、动作空间支持等方面有本质区别，对比学习能加深理解。
>     *   **混合方法**：许多现代算法（如SAC）融合了价值函数和策略优化，Q-Learning的原理是理解这些混合方法的前提。
> 
> #### **2\. 是否需要学习Actor-Critic框架？**
> 
> *   **必要性**：**是**，尤其是对PPO的理解至关重要。
> *   **原因**：
>     *   **PPO的核心依赖**：PPO属于Actor-Critic框架的改进，其中：
>         *   **Actor**：策略网络，负责生成动作。
>         *   **Critic**：价值网络，评估状态或动作的价值（用于计算优势函数 ( A\_t )）。
>     *   **优势函数的作用**：PPO通过Critic提供的优势函数 ( A\_t = Q(s,a) - V(s) ) 来指导策略更新，脱离Actor-Critic框架无法理解其设计动机。

  

  

  

Q-Learning
==========

看 huggingface 里面有比较详细的文档。 照着 huggingface 来学习。

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_19-45-35.png)

前面的 bellman 方程和四种价值函数里面， 我们知道了以下

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_19-48-45.png)

当前状态的价值是， 当前 Agent 在当前 State 开始的 discounted return, 然后根据 policy 进行行动得到的价值。

*   Policy-based training 中， 我们目的是找到一个最优的策略， 来最优化总价值函数， 总回报函数（而不是价值函数）
    *   我们不会手工定义 policy 的行为， 这是训练出来的
*   Value-based：我们简介的训练一个值函数（一个 state 或者 state-valued）的值， 该值函数输出状态和状态对的价值， 我们使用此值函数对应的值， 来找到 Action
    *   所以，策略是有的， 不过策略不是一个神经网络， 而是一个基于价值函数的策略： 输出的价值最高的 action 就是需要采用的 action（在确定性 action 的情况下）

  

有两个重要的函数：

### State-value function

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_20-15-18.png)

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_20-15-24.png)

按照 bellman 方程来说， 这个是可以应用动态规划的

  

Action-value function

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_20-17-28.png)

蒙特卡洛（Monte Carlo）和时间差异学习
------------------------

都是训练我们的价值函数或者我们的策略函数的。 他们都是基于「经验」来解决 RL 问题的方法。

区别在于:

*   蒙特卡洛：使用了整个经验
*   时间差：仅使用一步经验**(St,At,Rt+1,St+1St​,At​,Rt+1​,St+1​ )**

### **蒙特卡洛：learn at the end of the episode**

**蒙特卡洛在整个 episode 结束后， 使用其 Return 用作更新**

**![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_20-44-51.png)**

**对于起始位置的价值来说：![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_20-48-30.png)**

### **时间差异学习：learn at each step**

**也被称为 TD 学习。**

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_20-54-57.png)

直接用 Rt+1 来计算整体趋势。

  

Q-Learning
----------

是一种： **off-policy value-based method that uses a TD approach to train its action-value function：非严格遵循策略（off-policy）的， 基于时间差异的，来训练「状态价值」函数的价值函数方法**

**Q 代表是 Quality 的意思**

**在 Q-Learning 中， 价值函数被编码成了 Q 表。**

**开始后的时候都是 0 ， 之后在 episode 中， 每个 step 对齐进行更新。**

  

**Q-Learning Algorithm**
------------------------

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-8_21-17-54.png)

其实就是我们想象的那样，不过添加了一些细节：

*   初始化 Q-table，全都是 0 
*   使用 Epslon-greedy 策略（ɛ 的概率选择贪婪 Action， 其他时候随便选择）
*   执行动作， 获取到下一个状态的 Return 和 下一个状态（t+1）
*   更新 Q(t)
*   直到一个 episode 结束， 重复下一个 episode

  

  

(到此为止 2h40min 左右)

DQN
---

对上面的 Q-Learning Algorithm 来说， 可以完成一些 discrete and small（离散并且小规模）的任务（比如 FronzenLake-v1）， 对于 Atari games（应该是太空大战)，里面有 10^9 ~ 10^11 的状态。

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-9_10-38-43.png)

在大型状态空间环境中， Q-Table 的空间复杂度就比较大了， (因为其是 O(S \* A)）的。

[RL-ZOO](https://github.com/DLR-RM/rl-baselines3-zoo)，一个工具

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-9_10-47-28.png)

*   input：四帧画面
    *   预处理输入（降低状态的复杂度）
    *   reduce the size → 84 \* 84
    *   灰度（这个游戏里面颜色没有其他信息）
    *   裁剪不重要信息，然后将四帧堆叠起来
        *   堆叠的原因是有些信息一帧看不出来， 多帧就可以看出来（比如速度）
*   output：不同行动的 Q-Value

对于 Q 的更新， 无法直接应用于 Q-Table，在 DQN 里面， 创建了一个损失函数：**该函数比较了我们的预测的 Q-Value 和实际的 Q 值 Q-Target ， 并且使用梯度下降来更新 QN 来让我们更好的预测**

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-9_11-0-1.png)

The Deep Q-Learning training algorithm 有两个阶段：

*   Sampling（采样）：执行动作并将其观察值（经验）的 tuples 放在一个「replay memory」中
*   Training（训练）：「随机」选择一小部分 tuples， 梯度下降更新这些东西

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-9_11-5-56.png)

DNP 可能会遭受不稳定性的困扰（感觉是因为每个 step 更改 Q-Table 不会影响下一个 step），但是更新神经网络会影响所有情况的输出。

（所以， 上面有个 reset， 应该是先更新 Q， 每 C step 将角 Q 更新成 Q）。

为了增强训练的稳定性，采取了三种方法：

*   Experience Replay：经验重播， 更有效地利用体验
*   Fixed Q-Target：固定的角 Q， 来稳定训练
*   Double Deep Q-Learning：解决高度 Q 值问题

### 重播

*   Make more efficient use of the experiences during the training：在训练期间更有效的利用经验
    *   添加一个 buffer， 我们可以节省经验的样本， 以在培训期间重复利用
    *   可以多次从相同体验中学习
    *   （但是改变了 Q 以后， A 可能不是原来那个 A 了啊， 所以不是原来 A 就实际执行， 是原来 A 就用回放是么）
*   避免忘记以前的经验，并减少经验之间的相关性
    *   （是会导致不收敛吗？）

  

### 固定的 Q-Target 来稳定训练

当 Q 变化的太频繁， 会导致训练的显著震荡。

所以其中有个 C step 才更新一次的逻辑

  

### Double DQN（没深入理解）

计算 Q 时， 使用两个网络目标来计算 Q 值

使用 DQN 网络来计算 Action， 使用 Target network 来计算具体的 Q 值

  

（到这里近 4h）

策略优化
----

DS 没给出直接的资料， 但是看起来 [openai spinning up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) 里面有相关章节（换着学）

the theory of policy gradients 的三个关键结果：

*   [the simplest equation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient) describing the gradient of policy performance with respect to policy parameters,  
    [最简方程式](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient)描述了策略性能相对于策略参数的梯度，
*   a rule which allows us to [drop useless terms](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you) from that expression,  
    一条使我们能够从该表达式中[删除无用的术语](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you)的规则，
*   and a rule which allows us to [add useful terms](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients) to that expression.  
    以及使我们能够在该表达式中[添加有用术语](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients)的规则。

### 最简策略梯度

考虑随机，参数化策略

要使用梯度上升法优化回放函数， 就需要计算出其对应的梯度：

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_9-30-34.png)

一些对于分析梯度重要的 Fact

*   Probability of a Trajectory：轨迹的概率  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_9-31-23.png)  
    *   （这里没懂， P 不是自然法则吗， 最终算出来的应该是 ST 才对吧）（可能这里是 at 从 st → st+1 的概率（合理））
    *   所以等式左侧， 代表的是此轨迹以及参数的概率
*   The Log-Derivative Trick：对数导数 Trick  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_9-38-32.png)  
    *   微积分的简单规则：对于 x 来说， log x 的导数可以认为是 1/x（好像翻译的不对， 反正看公式带入就行， 数学原理暂时不用深究）
*   Log-Probability of a Trajectory：轨迹的对数概率  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_9-50-53.png)
*   Gradients of Environment Functions：环境函数的梯度，由于其中没有参数 /theta， 所以都为 0
*   Grad-Log-Prob of a Trajectory：因此轨迹的对数-Prob  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_9-53-0.png)
*   因此得到以下内容， 对于回报函数的积分就是：
    *   ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_9-53-9.png)
    *   这里终于理解了数学期望和里面的概率。
    *   所以最终梯度又变为了一个数学期望。

因为是数学期望， 所以可以用样本平均值来估算：

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_10-1-43.png)

收集一波轨迹， 然后计算其具体概率和奖励， 之后取平均， 就得到了我们想要的梯度。

（这里也有点不懂的， 随机概率的微分怎么实现呢？）

(4h40min)

一个最简的[实现源码](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py)，里面用了一些机器学习库，没有细读。

### Expected Grad-Log-Prob Lemma

期望梯度 Log-Prob 引理

![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_18-42-0.png)

假设 Ptheta 是变量为 x 时的参数化概率分布。

  

### Don't let the past distract you（没太看懂）

在上面公式里， 包含有一些噪声。

*   实际影响 return 的， 是执行 action 之后的 return， 而不是之前的。
*   所以在计算损失的时候用， 越往前的， 权重要加上后面的（好像是这个意思？）
*   ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_18-56-56.png)

  

（5h）

Actor-Critic 框架：
================

[https://huggingface.co/learn/deep-rl-course/unit6/introduction](https://huggingface.co/learn/deep-rl-course/unit6/introduction)，又回到 huggingface。 总体感觉 huggingface 的曲线更加柔和。

对于上面的基于策略的梯度下降法， 每个 episode 计算一次， 也就是用的蒙特卡洛计算方法。 导致：

*   we have significant variance in policy gradient estimation（我们有比较大的 gradient 差异）

策略梯度， 代表的是「回报最大的增长方向」。 并且由于需要用平均来估计数学期望（我理解是）， 需要用大量的样本来找到更加靠谱的数学期望。

  

Actor-Critic 是一种结合「Value」和「Policy」的方法，减少方差（波动），来稳定训练：

*   Actor: （演员）控制我们的 Agent 如何行动（Policy-Based method）
*   Critic：（评论家）我们行动的咋样（Value-Based method）

  

上面的训练过程在 huggingface 教程里也被成为 Reinforce。 一种理解是：

*   如果 return 很高， 我们希望加强这种轨迹的出现概率
*   如果 return 很低， 我们希望减弱这种轨迹的出现概率

  

  

### Advantage Actor-Critic(A2C)

主要实践的是 A2C。

*   一个初始 State
*   Actor 使用 Policy 得到一个 Action
*   Critic 使用此 State 以及 Action 计算出来 Q 值
*   执行 Action，得到 State(t+1) 和 R(t+1)
*   Actor 使用 Critic 已经给出的 Q 值更新其 Policy  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_19-39-56.png)  
    *   问题是， 这里更新该怎么更新啊？ 对于这个对数的求导值怎么得到。 而且含义是啥呢
*   Actor（更新后的） 给出 State(t+1) 对应 Action(t+1) 
*   Critic 更新其 value parameters  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_19-41-4.png)
    *   （文档没解释清楚）问了下 DS：Critic 的更新的目标是：  
        *   最小化 TD Error 来进行更新（评论家说的没错）
        *   ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_19-47-55.png)

### Adding Advantage ：

使用「优势函数」而不是 「Action-value 函数」来作为评论家。

优势函数的概念：

*   选择当前行动， 相比于其他行动好多少。  
    ![](./跟着 DS 学习 RL（二）经典算法分类与基本算法 - 原禄城 - Confluence_files/image-2025-2-10_20-7-31.png)

我们可以用 TD 误差， 来估计「优势函数」搞的好不好。