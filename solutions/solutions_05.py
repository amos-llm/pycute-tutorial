"""第 5 章练习参考"""

import os
import sys

# 参考解答放子目录, 把仓库根加进 sys.path 才能 import common。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycute import Layout, zipped_divide

from common import tile_to_shape

# 练习 1: 8 行 x 32 列的 atom 平铺到 (64,96): 8 个行 tile x 3 个列 tile
t = tile_to_shape(Layout((8, 32), (32, 1)), (64, 96))
# = ((8,8),(32,3)):((32,256),(1,2048))
# 步长来源: 行 tile 步长 = atom cosize = 8*32 = 256;
#           列 tile 步长 = 8 行 tile x 256 = 2048

# 练习 2: zipped_divide(mQ (200,128):(128,1), (128,128))
d = zipped_divide(Layout((200, 128), (128, 1)), (128, 128))
# = ((128,128),(2,1)):((128,1),(16384,0))  ->  size(d[0])=16384, d[1][0].stride=16384

# 练习 3: 行 = (8,16):(64,512), 列 = (64,2):(1,8192)
hand = Layout(((8, 16), (64, 2)), ((64, 512), (1, 8192)))
