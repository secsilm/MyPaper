# Unsupervised Deep Embedding for Clustering Analysis

## 3. Deep embedded clustering

考虑一个聚类问题，$n$ 个点 $\{x_i \in X\}$ 聚到 $k$ 个类别，每一类的中心用 $\mu_j$ 表示。和以往直接将原始数据空间 $X$ 中的点聚类不同的是，我们使用一个非线性映射 $f_\theta : X \to Z$，将原始数据映射到一个潜在特征空间 $Z$，而其中的 $\theta$ 是可学习的参数。为了避免维度诅咒，一般来说 $Z$ 的维度要低于 $X$。我们使用神经网络来训练得到 $\theta$。（*感觉这是一种将神经网络用于降维的想法，例如自编码器*）

该算法（DEC）能够在特征空间 $Z$ 中同时学习 $k$ 个聚类中心和非线性映射的参数 $\theta$。DEC有两个阶段：

1. 使用一个深度自编码器（deep encoder）进行参数初始化
2. 参数优化（即聚类），迭代计算一个辅助目标分布（auxiliary target distribution）和最小化 KL 散度

### 3.1 Clustering with KL divergence

给定一个初始化的非线性映射 $f_\theta$ 和初始化的类中心 $\mu_j$，我们使用一个非监督学习算法来提升聚类性能，主要分两步：

1. 在数据点（*特征空间 $Z$ 内的，即 embedded points*）和类中心之间计算一个软分配（soft assignment）
2. 更新 $\theta$，同时**使用辅助目标分布**根据当前高可信度的分配更新类中心（*应该就是将每一个点分配到概率最大的类中，然后更新类中心*）

这个过程一直持续到达到收敛标准。

#### 3.1.1 SOFT ASSIGNMENT

和 van der Maaten & Hinton 在 2008 年所用的一样，我们也使用 t 分布来衡量嵌入点 $z_i$ 和类中心 $\mu_j$ 之间的相似性：
$$q_{ij} = \dfrac{(1+\Vert{z_i-\mu_j}\Vert^2 / \alpha)^{-\dfrac{\alpha+1}{2}}}{\sum_{j'}(1+\Vert{z_i-\mu_{j'}}\Vert^2 / \alpha)^{-\dfrac{\alpha+1}{2}}}$$
其中 $z_i$ 是 $x_i$ 对应的嵌入点，即 $z_i = f_\theta(x_i)$，$\alpha$ 是 t 分布的自由度，$q_{ij}$ 可以被解释为样本 $i$ 分配到类别 $j$ 的概率（即软分配，soft assignment）。  
由于在非监督学习中我们不能通过交叉验证来求得 $\alpha$，而且根据 van der Maaten，学习 $\alpha$ 也是多余的，所以在所有实验中我们设置 $\alpha = 1$。

#### 3.1.2 KL DIVERGENCE MINIMIZATION

我们在辅助目标分布的帮助下通过学习高置信度的分配来调整类别。特别地，我们通过匹配软分配和目标分布来训练模型。为此，我们定义损失为软分配 $q_i$ 和辅助分布 $p_i$ 之间的 KL 散度：
$$L = KL(P \Vert Q) = \sum_i \sum_j p_{ij}log\dfrac{p_{ij}}{q_{ij}}$$
目标分布 $P$ 的选取对 DEC 的性能很关键。一个比较天真的做法是在一个置信度阈值上把每一个 $p_i$ 设置为 delta 分布（距离最近的类中心），并忽略其他样本（*应该指的是不在置信度阈值范围内*）。然而由于 $q_i$ 是软分配的，所以更自然的想法是使用软概率目标（sorter probability targets）。特别地，我们希望这个目标分布有如下特点：

1. 增强预测（即提高聚类纯度）
2. 更加重视那些以高置信度被分配的样本
3. 规范化每一个类中心的损失贡献以避免大的类扰乱潜在特征空间

在我们的实验中，我们通过计算 $q_i$ 的平方然后使用频数来规范化类的方法来计算 $p_i$：
$$p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{j'}q_{ij'}^2 / f_{j'}}$$
其中 $f_j = \sum_i q_{ij}$ 是软类别频数（soft cluster frequencies）。关于 $L$ 和 $P$ 的经验属性请参见 5.1 节的讨论。  

我们的训练策略可以被看成是一种自学习的形式（self-training）。在自学习中，我们有一个**初始化的分类器**和没有标签的数据集，**然后为了能够基于这个分类器自己的高置信度的预测，我们使用这个分类器来给数据集打上标签。实际上，在实验中我们发现 DEC 通过高置信度的预测和不断迭代提升了最初的初始化的值，这也返回来能够帮助低置信度的样本。**

#### 3.1.3 OPTIMIZATION

我们使用带动量的 SGD 算法联合优化类中心 $\mu_j$ 和 DNN 参数。$L$ 关于 $z_i$ 和 $\mu_j$ 的梯度分别按下式计算：
$$\frac{\partial L}{\partial z_i} = \frac{\alpha+1}{\alpha} \sum_j (1 + \frac{\Vert z_i - \mu_j \Vert^2}{\alpha})^{-1} \times (p_{ij} - q_{ij})(z_i - \mu_j)$$  
$$\frac{\partial L}{\partial z_i} = \frac{\alpha+1}{\alpha} \sum_i (1 + \frac{\Vert z_i - \mu_j \Vert^2}{\alpha})^{-1} \times (p_{ij} - q_{ij})(z_i - \mu_j)$$  
$\partial L / \partial z_i$ 然后被传给 DNN 用于在标准的反向传播中计算 $\partial L / \partial \theta$ 。为了发现类别分配，当连续两次迭代中有少于 $tol\%$ 的点改变类别分配的时候就停止程序。

### 3.2 Parameter initialization

到目前为止我们已经讨论了在给定 DNN 的参数 $\theta$ 和类中心 $\mu_j$ 的初始估计后 DEC 是如何执行的。现在我们来讨论这些参数和类中心是如何初始化的。

我们使用一个堆叠自编码器（stacked autoencoder，SAE）来初始化 DEC，因为最近的研究表明 SAE 能够在真实数据集中产生语义有意义的并且良好分离的表示。所以 SAE 学习到的非监督表示会促进 DEC 的类别表示的学习。

我们一层一层的初始化 SAE，每一层是一个去噪自编码器，通过在随机腐败（random corruption）后重建前一层的输出来训练。一个去噪自编码器是一个两层的神经网络，定义如下：
$$\tilde{x} \sim Dropout(x)$$
$$h = g_1(W_1\tilde{x} + b_1)$$
$$\tilde{h} \sim Dropout(h)$$
$$y = g_2(W_2\tilde{h} + b_2)$$
其中 $Dropout(\cdot)$ 是一个随机映射，它随机的选择一些输入将其置零，$g_1$ 和 $g_2$ 分别是编码器和解码器的激活函数，$\theta = \{W_1, b_1, W_2, b_2\}$ 是模型参数。训练目标是最小化最小二乘损失 $\Vert x - y \Vert^2_2$。在训练完一层后，我们使用他的输出 $h$ 作为输入来训练下一层。我们在所有的编码器/解码器对中使用 ReLU 激活函数，除了第一个编码器/解码器对的 $g_2$（因为它需要重建原始输入，而原始输入可能有正值和负值，例如均值为 0 的图像）和最后一个编码器/解码器对的 $g_1$（因为最终的数据嵌入需要保留所有信息）。

在逐层贪婪训练完成后，我们连接所有的编码器和解码器（编码器在前，解码器在后），以逐层训练相反的顺序，构建一个深度自编码器并微调重建损失。最终的结果是一个中间有一个瓶颈编码层（bottleneck coding layer）的多层深度自编码器。我们然后丢弃解码器层然后使用编码器层来做数据空间和特征空间的初始映射，如图 1 所示。
![fig1](http://i.imgur.com/uPhTb3Z.png)
为了初始化类中心，我们将数据传入初始化的 DNN 得到嵌入的数据，然后在特征空间 $Z$ 上运行 k-means 聚类算法得到 $k$ 个初始中心 $\{\mu_j\}^k_{j=1}$ 。