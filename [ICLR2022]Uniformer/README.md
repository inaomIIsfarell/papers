## **Uniformer：用于高效时空表征学习的统一transformer**



### **任务背景**

时空表征学习是视频理解领域的基本任务，有两个不同的挑战：

1. 视频内容包含大量的时空信息冗余，局部关联帧之间的目标动作关系是很微妙的；

2. 视频包含复杂的时空差异，因为跨长距离帧的目标关系是动态的



### **现有方法**

视频分类(Video Classification)领域目前前沿的方法有：3D卷积神经网络，spatiotemporal transformer（时空transformers），但是各有缺点：

1. 3D卷积受限于感受野大小，很难有效捕捉长距离的依赖；
2. 以往的spatiotemporal transformer（如TimeSformer等）在学习过程中会通过冗余的全局注意力来学习局部表征（such transformer learns local representations with redundant global attention）会浪费得大量的计算资源，做不到computation-accuracy balance



### **本文介绍内容**

为了解决这些问题，作者提出将 3D卷积 和 时空自注意力（spatiotemporal self-attention）以transformer的格式有效统一，即Uniformer，包含3个核心模块：**Dynamic Position Embedding**，多头关系聚合器**Multi-Head Relation Aggregator（MHRA）**，**Feed Forward Network**



uniformer和传统transformer最大的区别就在于关系聚合器

1. 区别于传统transformer在所有层中利用self-attention，本文提出的relation aggregator（**MHRA**）分开处理视频的冗余和依赖关系
   - 在浅层，aggregator通过一个小的可学习参数矩阵来学习局部关系，这可以通过聚合3D邻域中相邻tokens的上下文来极大降低计算负担；
   - 在深层，aggregator通过相似性对比来学习全局关系，这能灵活构建视频不同帧的长距离token的依赖关系

2. 与传统transformer分离时序attention与空间attention不同，**MHRA**在所有层共同编码时空上下文，这能通过联合学习的形式极大提高视频表征（our relation aggregator jointly encodes spatiotemporal context in all the layers, which can further boost video representations in a joint learning manner）



### **Uniformer**

总体结构如下图

![](/assets/overall_architecture_of_uniformer.jpg)
$$
X = DPE(X_{in}) + X_{in} {\quad} {\quad}(1)\\
Y = MHRA(Norm(X)) + X  {\quad}(2)\\
Z = FFN(Norm(Y)) + Y  {\quad}(3)\\ \\
X_{in} {\in} {\mathbb{R}^{C{\times}T{\times}H{\times}W}}
$$

- **MHRA**  (文中会将其成为**RA**)

  ​		将3D卷积和时空self-attention以transformer的形式结合，分别在浅层和深层解决视频冗余和依赖，通过多头融合（multi-head fusion）进行token关系学习
  $$
  R_{n}(\bf X) = A_nV_n(X){\quad}{\quad}(4)\\
  MHRA(\bf {X}) = Concat(R_1(\bf {X});R_2(\bf {X}); {\cdot}{\cdot}{\cdot} ;R_N(\bf {X}))U{\quad}{\quad}(5)
  $$
  

  ​		输入的tensor $\bf {X}_{in} {\in} {\mathbb{R}^{C{\times}T{\times}H{\times}W}}$，首先将其reshape为一个序列tokens $\bf{X_{in}} {\in} {\mathbb{R}^{L{\times}C }}$ ，其中 $L = T \times H \times W$

  

  $R_n({\cdot})$ 表示第$n$头的RA（the relation aggregator (RA) in the n-th head）

  $U {\in} \mathbb{R} ^ {C \times C}$ 是一个结合$N$头特征的可学习的参数矩阵

  经过一层线性变化，将原始token转化为上下文$V_n(\bf X) \in \mathbb{R} ^ {L{\times}{\dfrac{C}{N}}} $

  随后RA能通过$A_n \in \mathbb{R}^{L{\times}L}$ 对上下文进行有机聚合

  **RA**的关键就在于如何从视频中学习$A_n$

  

  - **Local MHRA**（局部 MHRA）

  ​        作者提出在浅层的目标就是从小的3D邻域中的局部时空上下文学习到视频表征的细节，这与3D卷积的设计有着一样的见解。浅层中，相邻tokens间视频内容变化很微妙，所以通过局部操作降低冗余以编码细节特征十分重要，因此在此作者将 $A_n$ 设计为在局部3D邻域中操作的可学习参数矩阵

  ​		给定一个 ${\bf X}_i$ ，**RA** 通过该token和同一邻域 $\Omega{_n}^{t \times h \times w}$ (见Appendix **A**)中其他tokens学习局部的时空信息
  $$
  A_n^{local}({\bf X}_i, {\bf X}_j) = a_n^{i-j}, {\quad} where {\quad} j \in \Omega _i ^ {t \times h \times w} \quad \quad (6)
  $$
  $a_n \in \mathbb{R} ^ {t \times h \times w}$ 表示可学习的参数矩阵， ${\bf X}_j$ 指的是邻域 $\Omega{_n}^{t \times h \times w}$ 中的任意相邻token，$(i - j)$ 为二者相对位置，表明 $A_n$ 的值只与相对位置有关 

  

  局部**MHRA**与3D卷积块对比，提出该局部**MHRA**和 MobileNet block风格类似，都是PWConv-DWConv-PWConv（提取特征，和常规卷积相比参数量和运算成本较低）

  

  - **Global MHRA**
  
    ​		在网络的深层，作者提出需要对整个特征空间建立长时关系（Inthedeeplayers, we focus on capturing long-term token dependency in the global video clip）和self-attention的思想一致，因此作者提出需要比较全局上下文相似度来构建 $A_n$
    $$
    A_n ^ {global} ({\bf X}_i, {\bf X}_j) = \frac{e^{Q_n({\bf X}_i)^T K_n({\bf X}_j)}}{{\sum}_{j ^ {'} \in {\Omega}_{T \times H \times W}}e^{Q_n({\bf X}_i)^T K_n({\bf X}_{j ^ {'}})}} \quad (7)
    $$
    ​		${\bf X}_j$ 是全局的3D邻域中的任意token，$Q(\cdot)$ 和 $K(\cdot)$  是不同的线性变换
  
    
  
    ​		先前的video transformer在所有阶段都用了self-attention，带来了很大的计算量，为了降低点乘带来的计算量，先前的工作采用时空注意力分离策略，但这显然损害了tokens的时空关系。本文提出的 **MHRA** 在浅层就对tokens的局部关系进行聚合节省了冗余计算量，网络在深层就可以使用对时空关系进行统一联合编码，得到更好的视频特征表达。
    
    
  
- **Dynamic Position Embedding**（动态位置编码）

​		视频具有时间和空间特性，对token的时空位置信息进行编码很有必要。之前的方法主要是在图片任务上进行绝对或相对位置编码。绝对位置编码在处理更大分辨率的输入是需要进行线性插值以及额外的参数微调，相对位置编码会对self-attention的形式进行修改且因为缺乏绝对位置信息使得模型表现更差，因此本文使用卷积位置编码设计**DPE**
$$
DPE({\bf X}_{in}) = DWConv({\bf X}_{in})
$$
​		$DWConv$ 就是零填充的深度可分离卷积。一方面，卷积对任何输入形式都很友好，也很容易拓展到空间维度统一编码时空位置信息。另一方面，深度可分离卷积十分轻量，额外的零填充可以帮助每个token确定自己的绝对位置。

​		

- **Model Architecture**

如上图所示，模型一共分4个stages，通道数分别是64，128，320，512。其中Uniformer-S {3, 4, 8, 3}，Uniformer-B {5, 8, 20, 7}

前两个stages使用局部**MHRA**，邻域大小设置为 $5 \times 5 \times 5$ ，头数 $N$ 与通道数对应，归一化使用BN；后两个stages使用全局**MHRA**，头维度设置为64，归一化使用LN

**DPE** 的卷积核大小为$3 \times 3 \times 3 \quad (T \times H \times W)$ ，**FFN** 的拓展倍数为4

作者还在stage1之前使用 $3 \times 4 \times 4$ 的卷积， $2 \times 4 \times 4$ 的步长对时空两个维度进行下采样，在其他stages前则使用$1 \times 2 \times 2 (conv) \quad 1 \times 2 \times 2 (stride)$ 的卷积进行下采样





