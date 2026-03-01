## 1.生成式AI是什么？

chatgpt在文字接龙时，会拆解成一连串文字接龙，这样在最终输出的时候就能穷尽无穷。



## 2.快速了解大模型

![image-20260204221706658](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260204221706658.png)

 gpt作为基石模型+微调->chatGPT

![image-20260204225244545](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260204225244545.png)

![image-20260204225355970](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260204225355970.png)

预训练GPT（穷尽的搜索最佳答案，每次进行拼接）+督导式训练->chatgpt（进行参与微调，**人类需要提供完整的正确答案**）->增强式，会咨询人类，第二次有没有比第一次好，如果有，就改变答案



增强式学习基本概念：**对齐人类的需求，根据人的反馈来提升答案几率**

1.台湾最高的山是那座？-> LLM -> 1.玉山 2.谁来告诉我  提高1的几率，降低2的几率

2.增强式学习步骤：a.模仿人类老师的喜好

![image-20260205075030363](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260205075030363.png)

b.向模拟老师学习

![image-20260205075146345](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260205075146345.png)



### 1.chatgpt使用技巧

a.把需求讲清楚

b.提供咨询给GPT（主要是因为他是文字接龙，所以需要输入准确的讯息）

c.提供示范(给个例子)

d.鼓励GPT想一想（最好列出详细过程，否则答案可能错误）



## 3.生成式人工智慧特别之处

![image-20260210073900980](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260210073900980.png)

### 1.如何让生成式语言模型答案更好

a.改变不了模型，那就改变自己（模型参数都是固定的）---prompt engineering,提供更好的prompt

b.训练自己的模型->开源模型，调整参数



## 4.训练不了人工智能，你可以训练你自己

![image-20260211233135048](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260211233135048.png)

1.可以通过输入参考示范提高答案准确性(in-context learning)

2.任务拆解，通过多个分解任务，形成最终答案

3.语言模型检查自己的任务输出是否正确（反省的过程是没有任何模型训练，函式是固定的，所以下次再次咨询时，还是会出错，反省才对）



### 1.如何提升大模型答案的正确率呢？

#### 1、方式一：组合拳

![image-20260215224921860](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260215224921860.png)

如上图所示：打一套组合拳，1 复杂的任务拆解为多个步骤执行+ 检查自己的答案+ 同一个问题多问几次（取答案出现多次的）

example:533215 * 251349等于多少，直接给出答案，这样结果往往是错误的



#### 2.方式二：使用工具-搜索引擎

![image-20260215225743563](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260215225743563.png)

#### 3.方式三：使用工具-写程式

![image-20260216153747978](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260216153747978.png)

#### 4.方式四：呼叫工具

![image-20260216154132784](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260216154132784.png)

#### 5.方式五：模型合作

![image-20260217175653626](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260217175653626.png)

但是要注意让模型停下来，需要合适的prompt（基于另外一个人的回答作为参考）

#### 6.方式六：引入不同的角色



## 5.大型语言模型修炼史

### 5.1 第一阶段，积累实力



![image-20260223174828436](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260223174828436.png)

机器学习有两个过程：1.训练找参数   2.测试（找到参数进行使用）



自督导式学习：根据文字资料进行文字接龙（所以需要进行资料清理）

![image-20260223224434559](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260223224434559.png)

### 5.2 第二阶段，名师指点发挥潜力

微调是将第一阶段的参数继续保留，加上新增的资料继续调优参数

![image-20260224073408695](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260224073408695.png)



预训练pre-train参数可以得到更正确的答案，有很强的举一反三能力

![image-20260224073814785](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260224073814785.png)

### 5.3 第三阶段，参与实战打磨机巧

1.预训练(pre-train),自督导式学习

2.调参(instruction fine-tuning)，督导式学习：人类需要想出问题，并且提供正确答案

3.RLHF（reinforcement learning），增强式学习：模型产生答案，人告知更好的答案，从而进行调参（算法PPO）

![image-20260225073402100](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260225073402100.png)

如果每个都让人类来回答，那成本太高了，引入**reward model(回馈模型)**来模仿人类喜好

![image-20260225215857828](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260225215857828.png)

**增强式学习的难题：遇到人类自己都无法正确判断好坏的情况，该如何选择？**

![image-20260225220523886](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260225220523886.png)

## 6.以大型语言模型打造的AI Agent

## 7.今日的语言模型是如何做文字接龙的（Transformer简介）

模型演进信息：

![image-20260227074147622](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260227074147622.png)



transformer概述：

![image-20260227074349989](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260227074349989.png)

### 1.Tokenization（把文字转化为token）

语言模型是以token为单位进行处理的，此时还没有要训练的参数，token是人为设定的

![image-20260227074617442](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260227074617442.png)

### 2.input layer(理解token)

embedding过程就是把一个事向量化，意思相近的token有相近的embedding，这样后面处理的时候，能知道这些token存在关联性

![image-20260228074400405](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260228074400405.png)

当前token embedding是在训练时得到的，没有考虑上下文

#### 2.理解每个token-位置

![image-20260228074729061](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260228074729061.png)

语言+位置都会embedding，每个位置都有positional embedding(**可以是人来设计，也可以是训练得到**)

### 3.Attention：考虑上下文

![image-20260228074856570](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260228074856570.png)

上图的两个过的embedding向量值是一样的，通过attention后embedding就会不一样，称为contextualized token embedding

attention的作用就是**将上下文的token咨询加到embedding里面**

![image-20260301223102222](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260301223102222.png)

找出相关的token：不同的token会计算相关性，相关性计算也是通过训练得到（参数）

![image-20260301223201037](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260301223201037.png)

集合相关咨询：根据attention weight计算出最终值

计算所有token两两间的相关性，得到attention matrix：

![image-20260301223419313](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260301223419313.png)

实际过程只会考虑左边（前面）的token

 ![image-20260301223642403](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260301223642403.png)

#### 1.当前模型用的都是multi-head attention

计算相关性有多种方式（看动物：猫和狗是相关的， 看狗：狗叫是相关的，所以计算相关性不能单一的只用一种方式），通常有16组计算相关性

![image-20260302073634057](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260302073634057.png)

#### 2.多个attention通过feed forward整合

将多个embedding 纵向整合形成一个进行输出(feed forward里面的参数也是训练得到)

#### 3.attention+feed forward=transformer block

#### 4.多个transformer block结合，最后将句尾的向量通过output layer（linear transform+softmax）得到几率分布

![image-20260302074252370](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260302074252370.png)

![image-20260302074412855](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260302074412855.png)

#### 5.语言模型产生答案过程总结

![image-20260302074847941](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260302074847941.png)

为什么处理超长文本是挑战？如果有100k token，那么attention计算是100k^100k(所有的token都需要两两进行计算embedding)，所以算力会是一个挑战

![image-20260302074956710](https://raw.githubusercontent.com/robertchb/newphoto/master/image-20260302074956710.png)