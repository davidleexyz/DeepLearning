### 后端优化
1. split 拆分
2. reorder 调换计算顺序
3. tile 平铺 (大的图像分割成块计算)
https://image.oldpan.me/lesson_05_tiled.gif
4. vectorize 向量化 (将计算的像素变成向量)
https://image.oldpan.me/lesson_05_vectors.gif
5. unrolling  展开 (将内层循环展开)

6. 融合tiling vectorize unrolling
https://image.oldpan.me/lesson_05_parallel_tiles.gif

#### 做一切优化的目的是为了 并行计算 加快执行速度

#### 两种后端优化思路:
1. tvm 采用 halide思想  将alogrithm与schedule 分离
schedule就是指手工优化经验, 因为halide是函数式的, 没有副作用,
所以计算顺序无关
2. tc, mlir 采用DSL 走polyhedral 的路线

#### 资源
https://ucbrise.github.io/cs294-ai-sys-sp19/#
