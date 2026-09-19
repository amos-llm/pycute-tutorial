"""第 9 章练习参考"""

from pycute import Layout, logical_divide
# 练习 1: 见章节 ex1 代码 (256 个偏移无重复且落在 [0,256))。

# 练习 2: C 碎片 (16x8) 行距 8:
your_V_C = Layout((2, 2), (1, 64))
# (col pair: 步长 1; row half +8: 步长 8*8 = 64)

# 练习 3: 完整重排流程 (真实配置: S 瓦片 (128,32), 128 线程, kNWarps=4):
acc_v = Layout(((2, 2), 2, 4), ((1, 256), 2048, 8))
div2 = logical_divide(acc_v, (None, None, 2))
# = ((2,2), 2, (2,2)):((1,256), 2048, (8,16))
# 重排: make_layout(make_layout(div2[0], div2[2][0]), div2[1], div2[2][1])
# = (((2,2),2),2,2):(((1,256),8),2048,16)  即 FA2 的 ((4,2), MMA_M, MMA_N/2)
