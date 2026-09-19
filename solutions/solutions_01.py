"""第 1 章练习参考"""

from pycute import Layout

# 练习 1: Layout((4,3),(3,1))(2,1) = 2*3 + 1*1 = 7
your_answer = 7

# 练习 2: (4,3):(1,4) 中偏移 7: 7 = i + 4j -> i=3, j=1
your_crd = (3, 1)

# 练习 3: 任意 size != cosize 的 layout, 例如:
your_layout = Layout((8, 4), (2, 1))  # size=32, 最大偏移 7*2+3=17, cosize=18
