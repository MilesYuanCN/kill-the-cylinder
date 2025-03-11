# Transformer

<details>
<summary>DS 提供的详细资料</summary>
Thinking...
> 
> 好的，我现在需要详细分析用户提供的关于大模型基础的学习大纲，并给出相应的学习资料和关键点。首先，用户的大纲分为两部分：Transformer速成（6小时）和模型压缩三剑客（6小时），每个部分下还有子主题。我需要为每个子主题推荐合适的学习资料，并列出关键点。
> 
> 首先处理第一部分：Transformer速成。核心组件包括自注意力机制、位置编码，以及对比RNN/CNN的优劣。用户可能希望快速掌握Transformer的基础，所以需要经典和易懂的资料。比如，原始论文《Attention Is All You Need》是必读的，但可能对新手来说有些难，所以需要补充教程或博客文章。李宏毅的视频课程可能适合视觉学习者，而Jay Alammar的博客图解形式更易于理解。
> 
> 对于自注意力机制，关键点包括Q/K/V的计算、多头注意力的作用，以及为什么自注意力能捕捉长距离依赖。需要确保理解矩阵运算和缩放点积注意力中的缩放原因（防止梯度消失）。
> 
> 位置编码部分，用户提到了绝对式和相对式。需要解释两者的区别，比如绝对式使用正弦函数，而相对式如Transformer-XL中的处理方式。关键点包括位置编码如何注入序列顺序信息，以及相对位置编码的优势，比如处理长序列时的泛化能力。
> 
> 对比RNN/CNN的优劣，重点在并行计算能力。RNN由于时间步依赖无法并行，而CNN的卷积核大小限制长距离依赖。这里需要强调Transformer在并行化和捕捉全局依赖方面的优势，同时也要提到计算资源的消耗问题。
> 
> 接下来是模型压缩部分，包括量化、蒸馏和剪枝。量化方面，INT8的原理和补偿方法（如校准）是关键。需要推荐实践性强的资料，比如TensorRT的文档或相关论文。蒸馏部分要关注如何将大模型知识迁移到小模型，流程包括教师-学生架构和损失函数设计。TinyBERT的论文和实现教程会有帮助。剪枝则涉及通道剪枝的方法，基于权重大小进行剪枝，需要了解如何评估剪枝后的模型性能。
> 
> 用户可能需要更多的实践指导，所以每个子主题的关键点应包含理论加应用。例如，在蒸馏中，除了流程，还要提到实际应用中的技巧，如数据增强和损失函数的选择。在量化部分，除了原理，还要考虑实际部署时的校准步骤和精度恢复方法。
> 
> 需要确保推荐的资料涵盖理论和代码实现，比如Hugging Face的Transformers库或者GitHub上的实战项目。同时，关键点要简明扼要，突出重点，方便用户快速复习。
> 
> 最后，检查是否有遗漏的子主题，比如模型压缩中的剪枝部分，是否涵盖权重幅度的通道剪枝，是否有推荐相关论文或工具（如NNI）。确保每个关键点对应学习资料的内容，帮助用户高效学习。
> 


以下是为您整理的学习大纲对应的详细学习资料与关键点，结合了理论、论文与实战资源：

---

### **一、Transformer速成（6小时）**
#### **1. 核心组件：自注意力机制**
- **学习资料**：
  - 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（原始Transformer论文，精读第3节）
  - 视频：[李宏毅-Transformer详解](https://www.bilibili.com/video/BV1J441137V6)（中文，通俗易懂）
  - 图解：[The Illustrated Transformer](https://arxiv.org/abs/1706.03762)（Jay Alammar博客，图文并茂）
- **关键点**：
  - **Query/Key/Value矩阵计算**：输入向量通过线性变换生成Q/K/V，计算注意力分数。
  - **缩放点积注意力**：`Softmax(QK^T/√d_k)V`，缩放因子√d_k防止梯度消失。
  - **多头注意力**：并行多个注意力头，提升模型对不同子空间特征的捕捉能力。
  - **自注意力的优势**：长距离依赖建模、全局上下文理解、并行计算。

#### **2. 位置编码（绝对式 vs 相对式）**
- **学习资料**：
  - 论文：[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)（相对位置编码）
  - 博客：[Positional Encoding in Transformers](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)（数学推导）
- **关键点**：
  - **绝对位置编码**：使用正弦/余弦函数（Transformer原版）或可学习嵌入（BERT）。
  - **相对位置编码**：建模元素间相对距离（如Transformer-XL中的位移注意力）。
  - **核心作用**：弥补自注意力机制对序列顺序的忽视，提升模型对位置的敏感性。

#### **3. 对比RNN/CNN的优劣**
- **学习资料**：
  - 论文：[Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)（Google官方解读）
  - 对比分析：[RNN vs CNN vs Transformer](https://towardsdatascience.com/transformers-vs-recurrent-neural-networks-895b01992a3b)
- **关键点**：
  - **RNN缺陷**：串行计算（无法并行）、长距离依赖梯度消失、训练速度慢。
  - **CNN缺陷**：局部感受野（需堆叠层数）、对长序列建模能力弱。
  - **Transformer优势**：全局注意力、并行计算、长程依赖捕捉；**缺点**：内存消耗大（序列长度平方复杂度）。
</details>




# [李宏毅-Transformer详解](https://www.bilibili.com/video/BV1J441137V6)
Bert 是 Unsupervised Trained 的 Transformer
Seq2seq model with "self-attention"
![alt text](image.png)
Self-attention 是个 layer。
RNN：循环神经网络
- 每个 output 都需要依赖所有 input
- 难以并行

CNN：卷积神经网络
- 每个 output 只需要依赖部分 input
- 但是单层 CNN 只能捕捉局部信息
- 需要堆叠多层 CNN 来捕捉全局信息
- 可以并行

Self-attention Layer
- 每个 output 依赖所有 input
- 可以并行
- 可以取代 RNN

![alt text](image-1.png)
Self-attention 流程：
- 拿每个 q 对每个 k 做 attention（q 点乘 k）得到 alpha 
- 对 attention score 做 softmax，得到 ^alpha
  - ![alt text](image-2.png)
- 拿 ^alpha 对 v 做加权，得到 output
  - ![alt text](image-3.png)

从上面最后一个图可以看出：
- b1 考虑了所有 input 的 a1, a2, a3, a4

为什么这个东西有助于矩阵运算
- 对于输入， a1, a2, a3, a4 是输入的向量
- 可以三次矩阵运算就计算出 q k v（都是向量）
  - ![alt text](image-4.png)
- 之后处理 q 和 k 的点乘， 可以一次矩阵运算就计算出 
  - ![alt text](image-8.png)
  - ![alt text](image-9.png)
- 之后的 softmax 和 v 的点乘， 可以一次矩阵运算就计算出 b1
- 总结
  - ![alt text](image-10.png)


### Multi-head Self-attention
- 把 input 切成多段，每段分别做 self-attention（即仅对于 a1 也有多个 q, k, v）
  - ![alt text](image-11.png)
- 最后获取到多个 b1, b2, b3, b4，将其通过一个矩阵运算，得到最终的 output
- 目的是：
  - 让模型捕捉到更多不同类型的特征

### Positional Encoding
- 因为 self-attention 没有考虑位置信息，所以需要加上位置信息
- 位置信息可以有多种方式，比如正弦函数、可学习嵌入
- 常常使用 pi 来表示位置信息， 其是一个向量， 其维度与 input 的维度相同，为一个 one-hot 向量
- 一般在运算中， 会将 pi 和 input 相加， 然后进行运算
- Wp 为手设的矩阵
  ![alt text](image-12.png)


## 如何将 Self-attention 应用到 Seq2seq model 中
Seq2seq model 的输入是序列，输出是序列

对于传统 RNN，其包含
- Encoder：将输入序列编码为固定长度的向量
- Decoder：将固定长度的向量解码为输出序列

使用 Transformer 来实现 Seq2seq model 时， 需要将 Encoder 和 Decoder 都使用 Transformer（self-attention）来实现

## 看一下经典 Transformer 的结构
![alt text](image-13.png)
- Encoder
  - Inputs 的 Embedding Layer 变成了 Vector
  - 加上 positional encoding
  - 经过 N 个 Encoder Layer，每个 Encoder Layer 包含四个 sub-layer：
    - Multi-head Self-attention
      - 就是上面讲的 self-attention， 其 seq2seq
    - Add & Norm
      - 将输入和输出相加，然后进行 layer normalization
        - layer normalization 是 对每个样本进行 normalization， 而不是对每个特征进行 normalization(batch normalization)
        - 使得平均值为 0， 方差为 1， 使得数据分布更加稳定
    - Feed Forward Network
      - 就是简单的全连接层，
      - 目的是引入非线性， 使得模型能够捕捉到更多的特征（Self-attention 只能捕捉到线性特征）
    - Add & Norm
      - 再次 ADD & Norm
- Decoder
  - 经过 N 个 Decoder Layer，每个 Decoder Layer 包含四个 sub-layer：
    - Masked Multi-head Self-attention
      - 在 Decoder 中， 需要使用 Masked Multi-head Self-attention， 其目的是防止 Decoder 看到未来的信息（也就是输入的信息）
    - Multi-head Self-attention
      - 这是 Cross-attention 层,其中:
        - Query (Q) 来自 Decoder 的输入
        - Key (K) 和 Value (V) 来自 Encoder 的输出
      - 这允许 Decoder 关注到 Encoder 处理的输入序列中的相关信息
    - Add & Norm
    - Feed Forward Network
      - 和 Encoder 中的 Feed Forward Network 类似

## Attention 的能力
- 在 self-attention 中， 可以建立起来不同 a 之间的联系
- 例如这个句子：
  - “The animal didn't cross the street because it was too tired”
  - 这个句子中， “it” 指代的是动物
  - ![alt text](image-16.png)
  - 这个关联度反应与 attention score 上， 上面我们提到， 其是 q 点乘 k 得到的一个维度为 1 的向量
- 在 tired 换为 wide 后， it 和 street 之间的关联度就会变得比较大。
- 这也是 Attention 更容易让人理解的原因



## 问题：
- 为什么叫 Attention 呢？
  - 因为其是基于注意力机制的， 其目的是让模型能够关注到重要的信息
  - 对于文本来说， 其注意力机制是基于词与词之间的相似度来计算的
  - 有了相似度， 就可以计算出注意力分数， 然后对注意力分数进行 softmax， 得到注意力分数的分布
- 计算 QKV 的 W 矩阵是如何得到的呢？
  - 是随机初始化的
  - 在训练过程中， 通过反向传播来更新 W 矩阵
  - 对于目前的大模型，参数量主要就是在 W 矩阵上


## 学习总结：
看完湾湾大佬的视频以后， 大概知道这个是什么了， 但是还是有很多东西比较模糊
- 在大模型中， 具体的输入和输出到底是什么
- Attention 到底为什么会 Attention， 这个 Attention 的意义是什么？ 感觉需要看看 CNN 和 RNN 
- 怎样理解 W 矩阵
- 对于 Multi-head 来说， 其到底是怎么做的？



# Attention Is All You Need
https://arxiv.org/abs/1706.03762

本论文提出了 Transformer 模型， 其主要特点是使用了 self-attention 机制， 并且其可以并行处理序列数据。
其中的实践：
- Encoder 中的 N = 6,
- Decoder 中的 N = 6,

## 训练：
主要是在翻译场景（英语到德语）
450 万个句子
8个 P100 训练了 12 小时

## 总结：
大致知道这个论文的价值了， 但是细节的疑问同上， 还得再看看


# illustrated transformer
https://jalammar.github.io/illustrated-transformer/

前面的知识都还一致， 从 Bringing The Tensors Into The Picture 开始

- 每个单词都被 embedding 到 512 维的向量中
![alt text](image-14.png)
  - 即 a1 a2 a3 这些 inputs, 由不同的 embedding 担任
- 之后每个单词都会流经编码器的两层中的每一层
  - ![alt text](image-15.png)
  - 在 self-attention 中， 这些路径存在依赖关系
  - 在 feed-forward 中， 这些路径是独立的
- self-attention at a high level
  - 使用矩阵， 来表示句子中的不同向量之间的关系
  - 例如， 对于 a1 来说， 其和 a2 的关联度为 0.8， 和 a3 的关联度为 0.2，（q 点乘 k 得到）
  - 对于 q1 k1 v1 来说， 其维度通常小于 a1 的维度， 大概是 64 维，而 a1 的维度是 512 维
  - 所以再回顾一下 Attention 的过程
    - 通过不同的矩阵， 从初始 embedding 后的向量， 计算出来 q1 k1 v1
      - q: query, 查询
      - k: key, 键
      - v: value, 值
    - 之后用 q 和 k 的点乘， 也就是查询和键的点乘， 得到一个两者之间的关联度 \alpha，也就是 attention score
    - 然后对 attention score 进行 softmax， 得到一个注意力分数的分布(类似于概率分布)，也就是 ^alpha
    - 之后对 ^alpha 对 v 进行加权， 得到 b1(点乘后求和， 也能变为矩阵运算)（也被称为 z1）
  - b1 的维度由于是 v 的加权和， 所以其维度与 v 的维度相同
  

## the beast with many heads
- 对于 multi-head 来说，有多个 wq wk wv 矩阵。
- 这些矩阵每个都是随机初始化的， 在训练过程中， 通过反向传播来更新这些矩阵
- 在训练之后， 每个集合用于将输入的 embeddings 转换到不同的空间中
  - 在上面的例子中， The animal didn't cross the street because it was too tired 这句话中， 不同的头， 考虑了不同的内容
  - 比如第一个头， 考虑了 it 和 animal 的关系， 第二个头， 考虑了 it 和 tired 的关系
  - 多了就不好理解了，但是比如说 it， 和 street 的关系就是不大， 在多个头里面， 可能就只有少数的头考虑了 it 和 street 的关系
    - ![alt text](image-18.png)

## Representing The Order of The Sequence Using Positional Encoding
- 上面一直没说位置信息， 我也一直没理解湾湾老师说的位置信息的价值在哪。
- Transformer 对于 x 进行了额外处理， 给每个 x 都加了一个位置信息，得到了新的 x1
  ![alt text](image-19.png)
- 注意， 这个是按位 +， 不改变 x 的维度。
- 加上的东西也比较神奇， 给它 plot 以下是这样的东西
  - ![alt text](image-20.png)


## Add & Norm
- feed-forward 的输入， 实际上不是 b1， 而是 b1 + x1
- 此处会有个疑问， b1 不是和 x1 的维度不同吗？ 为什么可以相加？
  - 因为是多头的， 一个 b1 的维度是 64， 8 个头就是 512。 其将多个头的 b1 拼接起来， 之后经历一个矩阵运算， 得到最终的 output
  - 对于此处的 W0 来说， 其的初始变量是随机初始化的， 在训练过程中， 通过反向传播来更新 W0
  - W0 主要为了融合多个头关注的不同空间的信息， 其也是训练出来的。
- 所以输入到 feed-forward 的实际上是 b1 + x1，还进行了一个 layer normalization
  - ![alt text](image-21.png)
  - 对于每个向量进行 layer normalization， 使得平均值为 0， 方差为 1
    - 减缓梯度消失/爆炸
    - 加速收敛
    - 使得数据分布更加稳定（起到正则化的作用）

## Feed Forward
- 就是简单的全连接层， 其目的是引入非线性， 使得模型能够捕捉到更多的特征（Self-attention 只能捕捉到线性特征）



再往后看， 就是多个编码器堆叠在一起， 每个编码器都包含 self-attention 和 feed-forward 两个 sub-layer，


## 解码器端
编码器最终输出了一组注意向量 k v， 这些向量会被送到解码器的每个位置上
![alt text](image-22.png)

对于每个解码器来说， 其包含三个 sub-layer：
- masked multi-head self-attention
  - 通过 masked 来防止解码器看到未来的信息
  - ![alt text](image-23.png)
  - 意味着， 即便是已经有了多个输出， 第一个输出， 其也看不到后面的输出
- multi-head cross-attention
  - 用解码器自己的 q 对编码器的 k v 进行 attention
  - 其他的计算方式和编码器类似
- feed-forward


### The Final Linear Layer And Softmax
经过多层解码器，最后输出一个一堆向量， 每个向量代表一个单词
每个向量是个浮点向量（因为经过了 Add & Norm），由 Linear 层和 softmax 层， 得到最终的输出。

Linear Layer 是一个全连接层， 其目的是将解码器产生的向量投影到一个更大的向量空间中， 称为 logits。
如果模型在训练数据集中学习到了 10000 个单词， 那么 logits 的维度就是 10000， 每个单元格代表一个唯一单词的分数。
之后， softmax 层将这些分数转换为概率， 概率最高的单元格对应的单词就是模型的预测结果。


## 对于训练
### The Loss Function 损失函数
我们会有训练集， 其会包含很多个句子， 每个句子都会有一个正确的翻译。
我们可以让模型输出一个翻译， 然后计算其和正确翻译之间的差异， 这个差异就是损失。
这个差异的两种计算方式：
- 交叉熵损失（Cross-Entropy Loss）
- Kullback-Leibler Divergence（KL 散度）


## 总结
这个文章是真不错， 比视频和论文讲了更多的要点。 还有更多关联的文章。




## 问题
- 不同的头， 考虑了不同的内容， 但是因为他们都是随机的， 会有几个头考虑的空间是相同的吗？
  - ![alt text](image-17.png)
  - （用 cursor 做笔记是真的舒服， 这里也豁然开朗了， 不知道后面所说的剪枝是不是也是这个意思）
- 对 feedforward 不太懂
  - 其每个输出会与所有的输入相关联。
- 解码器的输入矩阵大小是不断变化的吗？
  - 是不断变化的， 因为其需要考虑已经输出的内容
  - 一开始是 <start>， 之后是 <start> <translate>， 最后是 <start> <translate> <to> <english> <end>
  - 这也会使得解码器输出的越多， 下一个单词的输入矩阵越大， 计算量越大的情况。
  
















