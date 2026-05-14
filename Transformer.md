# Transformer架构

## 0.transformer原理简介

举例：用电电电鳗电鳗会不会电死

### 1.输入层

![image-20260507072959283](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507072959283.png)

1.接着将每个token通过embedding转化为512维

![image-20260507073054643](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073054643.png)

2.不同位置的字也得记录，添加位置编码。位置编码+词向量

![image-20260507073225266](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073225266.png)

### 2.注意力机制

#### 2.1 单头注意力

![image-20260507073356382](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073356382.png)

![image-20260507073410661](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073410661.png)

得到10个Q,K,V的值。

Q：当前词想要关注什么？

K：能为其他词提供什么？

V：词包含的实际信息

![image-20260507073730588](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073730588.png)

接着通过softmax进行归一化*v

![image-20260507073826633](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073826633.png)

最后每一行进行相加，得到最终的V1-V10

![image-20260507073932139](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507073932139.png)

最后得到向量z，但是字和字之间的语义并不能理解，如下图所示

![image-20260507074304737](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507074304737.png)





#### 2.2 多头注意力机制

将10乘512->8乘10乘64

 ![image-20260507074224338](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507074224338.png)

![image-20260507074349817](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507074349817.png)

![image-20260507074407517](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507074407517.png)

最后进行加权得到最终的v1-v10

### 3.残差连接+归一化+前馈网络

残差网络就是防止每一层之间输入值一样，导致计算结果一样，F(N) = F(X) + b(b就是残差)，归一化将值整合在0-1之间，FNN通过ReLu(A)将原来直线通过基函数变成曲线函数，将语义丰富起来

![image-20260507074548248](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507074548248.png)



![image-20260507074801686](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507074801686.png)



![image-20260507075036047](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260507075036047.png)

![image-20260508072715946](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260508072715946.png)

引入掩码多头自注意力机制，将输入的后文内容遮盖掉

![image-20260508073033952](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260508073033952.png)

交叉注意力机制作用实现前面输入和后面输出的对齐，不和自己计算，和前文说的注意力机制差异就在不和自己计算。

### 4.输出层



![image-20260508073356609](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260508073356609.png)



## 1.注意力机制

### 1.1 什么是注意力机制

从 计算机视觉（Computer Vision，CV）为起源发展起来的神经网络，其核心架构有三种：

- 前馈神经网络(feedforward neural network,FNN)，从输入层单向流动到输出层，无循环结构，各层之间通过全链接或特定方式传递信息，其中多层感知是最常见的形式，每一层都和上下两层的每一个神经完全相连

  ![image-20260419092634632](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260419092634632.png)

FNN也是transformer的计算引擎，如下图所示，transformer中的encoder-decoder图，FNN关键作用（**对自注意力输出进行非线性特征变换，与自注意力分工协作提升模型性能**）：

![image-20260419092733533](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260419092733533.png)

FNN核心结构与数学原理：类似数据加工厂流水线--数据输入（原材料） -> 升维线性变换（拆解分类原材料） ->非线性激活（加工赋予特殊属性） -> 降维线性变换（组装成标准化产品）

FNN(x) = max(0, xW1+b1)W2 + b2

为什么需要升维？ 一个句子的情绪/神态等通过升维来理解

为什么要升维再降维？ 方便残差连接

**FNN参数量占比**：FNN通常占据参数量的60%-80%，是模型表达能力核心支撑部分，负责对自注意力机制输出的上下文表示非线性变换和特征提取。



- 卷积神经网络(CNN)训练参数量远小于全连接神经网络的卷积层来进行特征提取和学习

![image-20260419174515830](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260419174515830.png)

CNN是图形识别和分类的核心技术，有强大的特征提取能力：4*4的矩阵，可以根据2*2得到一个卷积核，从而缩小大小，降低参数量

![image-20260419175656915](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260419175656915.png)

- 循环神经网络（RNN）：使用历史信息作为输入、包含环和自重复的网络。带着[记忆包]处理序列，感知时间顺序

  ![image-20260419181552662](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260419181552662.png)

ht= tanh(Wx'*'xt + Wh'*'ht-1+b)

ht = 更新后的新记忆

tanh()=激活函数：把数值压进安全范围

Wx,xt= 当前词向量及权重

Wh,ht-1旧记忆：上一时刻记忆及其权重

b偏置:可学习的基准调节项



RNN的缺点：

1.序列依序计算模拟时序信息，但是限制了计算机并行计算能力

2.RNN难以捕捉长序列相关关系



什么是注意力机制？

注意力机制最先源于计算机视觉领域，其核心思想为当我们关注一张图片，我们往往无需看清楚全部内容而仅将注意力集中在重点部分即可。而在自然语言处理领域，我们往往也可以通过将**重点注意力集中在一个或几个 token**，从而取得更高效高质的计算效果。

注意力机制有三个核心变量：**Query**（查询值）、**Key**（键值）和 **Value**（真值）。我们可以通过一个案例来理解每一个变量所代表的含义。例如，当我们有一篇新闻报道，我们想要找到这个报道的时间，那么，我们的 Query 可以是类似于“时间”、“日期”一类的向量（为了便于理解，此处使用文本来表示，但其实际是稠密的向量），Key 和 Value 会是整个文本。通过对 Query 和 Key 进行运算我们可以得到一个权重，这个权重其实反映了从 Query 出发，对文本每一个 token 应该分布的注意力相对大小。通过把权重和 Value 进行运算，得到的最后结果就是从 Query 出发计算整个文本注意力得到的结果。



### 1.2 深入理解注意力机制

注意力机制的核心变量：Query、key、value，以字典为例子，逐步分析下注意力机制的计算公式如何得到：

```
{
    "apple":10,
    "banana":5,
    "chair":2
}
```

字典的键就是注意力机制中的键值 Key，而字典的值就是真值 Value。

当我们想查找一个包含多个key的概念呢？比如需要查找'fruit'，当我们Query的'fruit'，我们可以分别给三个key赋予如下权重：

```
{
    "apple":0.6,
    "banana":0.4,
    "chair":0
}
```

那么value=0.6∗10+0.4∗5+0∗2=8

给不同 Key 所赋予的不同权重，就是我们所说的**注意力分数**,但是，如何针对每一个 Query，计算出对应的注意力分数呢？

我们往往用欧式距离来衡量词向量的相似性，但我们同样也可以用点积来进行度量：
$$
v·w=∑viwi
$$
根据向量定义，语义相似两个词点积大于0，语义不相似点积小于0。

假设Query为'fruit'，对应词向量为q;key对应的词向量为k=[VappleVbananaVchair]，这样可以计算Query和每一个键相似程度：
$$
x=qK^T
$$
接着需要通过Softmax将其转化为1的权重：
$$
softmax(x)_i=e_{xi}/∑_je_{xj}
$$
这样，得到的向量就能够反映 Query 和每一个 Key 的相似程度，同时又相加权重为 1，也就是我们的注意力分数了。

此时的值还是一个标量，同时，我们此次只查询了一个 Query。我们可以将值转化为维度为 dv 的向量，同时一次性查询多个 Query，同样将多个 Query 对应的词向量堆叠在一起形成矩阵 Q，得到公式：
$$
attention(Q,K,V)=softmax(QK^T)V
$$
我们离标准的注意力机制公式还差最后一步。在上一个公式中，如果 Q 和 K 对应的维度 dk 比较大，softmax 放缩时就非常容易受影响，使不同值之间的差异较大，从而影响梯度的稳定性。因此，我们要将 Q 和 K 乘积的结果做一个放缩：
$$
attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V
$$

### 1.3 自注意力

注意力机制的本质是对**两段序列的元素依次进行相似度计算**，寻找出一个序列的每个元素对另一个序列的每个元素的相关度，然后基于相关度进行加权，即分配注意力。

计算相似度：Q*K^T

权重化归一：
$$
softmax(QK^T)
$$
加权输出：
$$
softmax(QK^T)*V
$$
在 Transformer 的 Encoder 结构中，使用的是 注意力机制的变种 —— **自注意力**（self-attention，自注意力）机制。即是计算本身序列中每个元素对其他元素的注意力分布，即在计算过程中，**Q、K、V** 都由**同一个输入通过不同的参数矩阵计算得到**。在 Encoder 中，Q、K、V 分别是输入对参数矩阵 Wq、Wk、Wv 做积得到，从而拟合输入语句中**每一个 token 对其他所有 token 的关系**。

### 1.4 掩码自注意力(Mask self-attention)

注意力模型是生成模型，生成单子是一个一个生成的

eg:i have a dream

i->i have->i have a->i have a dream->i have a dream<eos>

掩码自注意力机制解决上述问题，可以**每一行输入中，模型仍然是只看到前面的 token，预测下一个 token。但是注意，上述输入不再是串行的过程，而可以一起并行地输入到模型中，模型只需要每一个样本根据未被遮蔽的 token 来预测下一个 token 即可**

```
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>
```

掩码注意力公式：
$$
softmax(QK^T + Mask)*V
$$
其中
$$
Mask_{ij} = \begin{cases}
0 \quad & j\le i \quad \text{（当前/前文，保留）}\\
-\infty \quad & j>i \quad \text{（未来词，屏蔽）}
\end{cases}
$$

### 1.5 多头注意力

1.通过上述注意力，给定一个X，通过注意力模型，得到一个Z，这个Z就是对X的新表征（词向量），Z这个词向量相比较X拥有了句法特征和语义特征。

2.Z相对X有了提升，通过Multi-Head Self-Attention，得到的Z‘相比较Z又有了进一步提升 ，多头的个数用h表示，一般h=8



多头注意力如何得到？

  对于X，将X分成了8块（8头），得到Z0-Z7，然后把Z0-Z7拼接起来，再做一次线性变换（改变维度）得到Z

多头注意力公式：
$$
MultiHead(Q,K,V) = \mathrm{Concat}(head_1,...,head_h)W^O \quad \text{where} \quad head_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$


## 2. Encoder-Decoder

Transformer 由 Encoder 和 Decoder 组成，每一个 Encoder（Decoder）又由 6个 Encoder（Decoder）Layer 组成。输入源序列会进入 Encoder 进行编码，到 Encoder Layer 的最顶层再将编码结果输出给 Decoder Layer 的每一层，通过 Decoder 解码后就可以得到输出目标序列了。

### 2.1 前馈神经网络FNN

  每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。

- FNN位置：自注意力->Add&Norm（残差连接与归一化）->FNN->Add&Norm经典结构

![image-20260427074148611](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260427074148611.png)

- FNN核心功能：对注意力输出进行非线性特征变换，与注意力分工协作提升模型能力（升维(512->2048维度)->非线性(RELU函数)->降维[2048->512]逐位置操作）

$$
FNN(x) = max(0, xW1+b1)W2 + b2
$$

- FNN在transformer中的关键作用（**在典型的transformer架构中，FNN占据总参数量的60%~80%**）：

  1.特征增强：拆解语法结构（升维）-> 理解深层含义(非线性变换)->重组为流畅表达(降维)

  2.机制互补：自注意力机制全局依赖，FNN精修局部特征（这道菜太辣了，FNN可以感受辣这个字，注意力只能感知菜->辣）

  3.并行计算

  4.训练稳定：输出维度匹配输入，为残差连接与层归一化提供基础

### 2.2 层归一化（layer norm）

​	神经网主流的归一化一般有两种，批归一化和层归一化

​	归一化核心是为了让不同层输入的取值范围或者分布能够比较一致。由于深度神经网络中每一层的输入都是上一层的输出，因此多层传递下，对网络中较高的层，之前的所有神经层的参数变化会导致其输入的分布发生较大的改变。也就是说，随着神经网络参数的更新，各层的输出分布是不相同的，且差异会随着网络深度的增大而增大。但是，需要预测的条件分布始终是相同的，从而也就造成了预测的误差。

#### 2.3.1 拟归一化

​	拟归一化：针对同一维度（同一特征）在所有样本（batch）中进行标准化。例如，假设特征维度包括“体重”，“身高”，“成绩”等，BN会分别对所有样本的同一特征计算均值和方差，再进行归一化处理。

​	BN的优点：1.环节内部协方差偏移：通过稳定层（奶茶师傅通过标准杯，放入固定量的珍珠等，加速奶茶制作）间输入分布，降低学习难度。  2.缓解梯度饱和问题（抹茶放到很多会进入饱和区，所以需要再白水里面放入抹茶，进行测试验证口味）：改善神经元激活函数的梯度分布，加速收敛  3.优化损失平面：通过平滑损失函数曲面提升收敛速度

​	BN缺点：对小batchsize敏感。 1.问题根源：BN依赖batch内样本均值和方差估计整体分布。batchsize过小，均值和方差无法代表全局分布  2.实例说明：a.全班52人，单个batch仅包含10人，其“体重”的均值和方差可能偏离全班真实分布。 b.极端情况下（batchsize=1），归一化失去意义

​	

#### 2.3.2 层归一化

​	相较于 Batch Norm 在每一层统计所有样本的均值和方差，Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计量的维度不同。

​	LN基于隐藏层的维度求均值和方差，对每一句话中的**每个token的embedding求方差和均值**，并不是针对整个句子计算的

​	每个位置的token对应的权重矩阵是不同的，所以这里肯定不能对不同矩阵的参数柔和到一起做归一化。



transformer为什么使用LN，而不是BN？

eg:我爱中国共产党

​     今天天气真不错 ， 如果使用BN(竖着计算方差和均值)毫无关联，但是使用LN（横着计算方差和均值）是有关联的



### 2.3 残差连接

​	由于 Transformer 模型结构较复杂、层数较深，为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。

​	例如，在 Encoder 中，在第一个子层，输入会先进行层归一化（Layer Norm），然后进入多头自注意力层，其输出会与原输入相加。在第二个子层也是一样。即：
$$
x = x + \text{MultiHeadSelfAttention}(\text{LayerNorm}(x))
$$

$$
\text{output} = x + \text{FNN}(\text{LayerNorm}(x))
$$

### 2.4 Encoder

​	实现上述组件后，就可以搭建一个Transformer的Ecoder。Encoder 由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。

### 2.5 Decoder

​	类似的，我们也可以先搭建 Decoder Layer，再将 N 个 Decoder Layer 组装为 Decoder。但是和 Encoder 不同的是，Decoder 由**两个注意力层**和**一个前馈神经网络**组成。第一个注意力层是一个掩码自注意力层，即使用 Mask 的注意力计算，保证每一个 token 只能使用该 token 之前的注意力分数；第二个注意力层是一个多头注意力层，该层将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。最后，再经过前馈神经网络。



## 3.搭建Transformer

#### 3.1 Embedding层

​	Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表。也就是说，在输入神经网络之前，我们往往会先让自然语言输入通过分词器 tokenizer，分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index。例如，如果我们将词表大小设为 4，输入“我喜欢你”，那么，分词器可以将输入转化成：

```
input: 我
output: 0

input: 喜欢
output: 1

input：你
output: 2
```

​	当然，在实际情况下，tokenizer 的工作会比这更复杂。例如，分词有多种不同的方式，可以切分成词、切分成子词、切分成字符等，而词表大小则往往高达数万数十万。此处我们不赘述 tokenizer 的详细情况，在后文会详细介绍大模型的 tokenizer 是如何运行和训练的。

​	因此，Embedding 层的输入往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。例如，对上述输入，Embedding 层的输入会是：

```
[[[0],[1],[2]]]
```

​	其 batch_size 为1，seq_len 为3，转化出来的 index 如上。

​	而 Embedding 内部其实是一个可训练的（Vocab_size，embedding_dim）的权重矩阵，词表里的每一个值，都对应一行维度为 embedding_dim 的向量。对于输入的值，会对应到这个词向量，然后拼接成（batch_size，seq_len，embedding_dim）的矩阵输出。

![image-20260509073525996](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260509073525996.png)

![image-20260509073607989](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260509073607989.png)



#### 3.2 位置编码

​	位置编码，即根据序列中 token 的相对位置对其进行编码，再将位置编码加入词向量编码中。位置编码的方式有很多，Transformer 使用了正余弦函数来进行位置编码（绝对位置编码Sinusoidal），其编码方式为：
$$
\begin{cases}
\text{PE}(pos, 2i) = \sin\left( pos \big/ 10000^{2i / d_{\text{model}}} \right) \\[6pt]
\text{PE}(pos, 2i+1) = \cos\left( pos \big/ 10000^{2i / d_{\text{model}}} \right)
\end{cases}
$$
​	上式中，pos 为 token 在句子中的位置，2i 和 2i+1 则指示了位置编码向量的维度索引是奇数还是偶数，从上式中我们可以看出对于奇数维度和偶数维度，Transformer 采用了不同的函数进行编码。

​	我们以一个简单的例子来说明位置编码的计算过程：假如我们输入的是一个长度为 4 的句子"I like to code"，我们可以得到下面的词向量矩阵 x ，其中每一行代表的就是一个词向量， x0=[0.1,0.2,0.3,0.4] 对应的就是“I”的词向量，它的pos就是为0，以此类推，第二行代表的是“like”的词向量，它的pos就是1：
$$
X = \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.2 & 0.3 & 0.4 & 0.5 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.4 & 0.5 & 0.6 & 0.7
\end{bmatrix}
$$
则经过位置编码后的词向量为：
$$
X_{\text{PE}} =
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.2 & 0.3 & 0.4 & 0.5 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.4 & 0.5 & 0.6 & 0.7
\end{bmatrix}
+
\begin{bmatrix}
\sin\left(\frac{0}{10000^0}\right) & \cos\left(\frac{0}{10000^0}\right) & \sin\left(\frac{0}{10000^{2/4}}\right) & \cos\left(\frac{0}{10000^{2/4}}\right) \\
\sin\left(\frac{1}{10000^0}\right) & \cos\left(\frac{1}{10000^0}\right) & \sin\left(\frac{1}{10000^{2/4}}\right) & \cos\left(\frac{1}{10000^{2/4}}\right) \\
\sin\left(\frac{2}{10000^0}\right) & \cos\left(\frac{2}{10000^0}\right) & \sin\left(\frac{2}{10000^{2/4}}\right) & \cos\left(\frac{2}{10000^{2/4}}\right) \\
\sin\left(\frac{3}{10000^0}\right) & \cos\left(\frac{3}{10000^0}\right) & \sin\left(\frac{3}{10000^{2/4}}\right) & \cos\left(\frac{3}{10000^{2/4}}\right)
\end{bmatrix}
=
\begin{bmatrix}
0.1 & 1.2 & 0.3 & 1.4 \\
1.041 & 0.84 & 0.41 & 1.49 \\
1.209 & -0.016 & 0.52 & 1.59 \\
0.541 & -0.489 & 0.895 & 1.655
\end{bmatrix}
$$
