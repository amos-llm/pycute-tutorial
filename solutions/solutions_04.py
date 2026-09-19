"""第 4 章练习参考"""

from pycute import composition, Layout

# 练习 1: R = composition(A, B) = (2,3):(2,4)
# 推导: A 的 1D 索引 i -> (i%3)*4 + i//3 (列主序分解); 代入 i = 6p+q (q<3):
#   i%3 = q, i//3 = 2p  ->  A(6p+q) = 4q + 2p, 即 p 步长 2 (=6//3), q 步长 4
R = composition(Layout((3, 4), (4, 1)), Layout((2, 3), (6, 1)))

# 练习 2: 取 A2 1D 序列的第 0,2,4,6 个元素 = 0,8,16,24 -> (4):(8)
your_answer = Layout(4, 8)

# 练习 3: pycute 逐个检查 A 的非末 mode: 非末 mode 的 shape 是 3,
# 与 stride 2 互不整除 (3%2≠0 且 2%3≠0) -> assert (末 mode 4:1 不被检查)。
# 注意: 这不是 "复合本身无定义" —— C++ 里 (3,4):(4,1) o (2):(2) 有定义且 = 2:8
# (B 只取 0, 8 两个元素, 都落在 A 第 0 行); pycute 只是 assert 更严。
