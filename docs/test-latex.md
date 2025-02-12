# LaTeX 公式测试

## 行内公式测试

在强化学习中，策略梯度可以表示为 $\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot A(s,a)]$。

## 块级公式测试

PPO的目标函数是：

$$
L^{CLIP} = \mathbb{E}\left[ \min\left( r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \right) \right]
$$

其中，概率比 $r(\theta)$ 定义为：

$$
r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
$$

## 复杂公式测试

贝尔曼方程：

$$
V^\pi(s) = \mathbb{E}_{a\sim\pi}\left[R(s,a) + \gamma \mathbb{E}_{s'\sim P}\left[V^\pi(s')\right]\right]
$$

优势函数：

$$
A(s,a) = Q(s,a) - V(s)
$$ 