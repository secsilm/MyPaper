# 聚类算法性能评价方法

一个聚类算法的性能评价不像监督算法那样简单，监督算法（如分类算法）的评价只是计算被错误分类的样本数量或者查准率（precision）或者回召率（recall）。特别的，任何评价方法不应该只考虑类别标签，因为聚类算法的目的更多是使得相似的点聚在一起，不相似的点散开，类内离散度最小，而类间离散度最大。

## Adjusted Rand Index（ARI）

> [sklearn.metrics.adjusted_rand_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score)

### 特点

- 函数：`sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)`
- 需要知道样本真实标签 `labels_true` 
- 忽略排列（permutation）和机会规范化（with chance normalization）
- 具有对称性
- 范围：[-1, 1]
- 随机标签分配的 ARI 接近于 0
- 对类的结果没有假设

### 数学公式

## 基于 Mutual Information 的方法

## Homogeneity，completeness 和 V-measure

## Fowlkes-Mallows scores

## Silhouette Coefficient

### 特点

> [sklearn.metrics.silhouette_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

- 函数：`sklearn.metrics.silhouette_score(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds)`
- **不需要知道样本真实标签**
- 范围：[-1, 1]
- 当集群被很好的分离并且比较密集时，该值会比较高

### 数学公式

一个样本集合的 Silhouette Coefficient 是集合中所有样本的 Silhouette Coefficient 的和的平均值。对于集合中的一个样本，对应的 Silhouette Coefficient 如下式计算：
$$ s = \frac{b-a}{max(a,b)} $$
其中，
- $a$： 一个样本到和其同一类的所有样本的平均距离
- $b$：一个样本到其*下一个最近类* 中的所有样本的平均距离

## Calinski-Harabaz Index
