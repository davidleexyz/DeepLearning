## 量化相关
### 如何提升量化后的模型精度
- 使用参数较多的模型 (例如resnet的8bit比mobilenet的8bit 精度下降的少)
- 使用不同的量化策略 
  - 如果数据分布比较集中, 采用asym要比sym方式更好, 因为非对称的量化粒度更小
  - 饱和方式比不饱和方式好, 可以忽略一些离群点
  - SQNR 采用per-channel 的方式 信噪比 要比per-layer好
- simulated quantization (训练的方式)
  利用反向传播建模量化噪声对模型的影响, 使weight的分布更加接近量化后的分布<br>
  forward pass: <br>
    得到的定点值转成浮点值, 带入量化误差<br>
  backward pass: <br>
    按照原来的梯度传播公式 <br>

### 精度提升总结[4]：
- 按照每个卷积核（axis，channel）进行量化，同时使用asymmetric来降低量化精度可以在不修改模型的前提下获得最高的精度
- Activation在8-bit情况下任然没有模型的精度下降，这样应该是由于BN和RELU6这种限定output范围的op的作用
- 拥有更多冗余参数的模型如inception在量化后可以获得比精简模型更好的精度 <br>
  Weight在量化过程中造成的影响是最大的
- 对于参数较少的模型如mobilenet，应使用线性量化而非饱和式样量化，因为在较少的参数意味着较大的权重系数，这样饱和量化对于离群点的引向就很显著。如高通的量化框架就申明对于mobilnetv1的精度下降很多
