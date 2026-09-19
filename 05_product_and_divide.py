"""第 5 章: 乘与除 —— product / divide (平铺与切块)

目标: 掌握 FA2 里另外两个高频操作:
  - "product" (乘): 把一个小 atom 平铺成大图块   -> FA2 的 SmemLayoutQ 等
  - "divide"  (除): 把大图块按 tile 切开          -> FA2 的 local_tile 等

代数定义 (02_layout_algebra.md):
  logical_product(A, B) = "按 B 的形状把 A 重复一遍"
  logical_divide(A, B)  = A o (B, complement(B, size(A)))
                          = (tile 内部布局, 第几个 tile)

运行: python3 05_product_and_divide.py
"""

from common import (
    assert_same,
    banner,
    check,
    domain_2d,
    exercise,
    step,
    tile_to_shape,
)
from pycute import (
    Layout,
    logical_divide,
    logical_product,
    size,
    tiled_product,
    zipped_divide,
)

banner("第 5 章: 乘与除 —— product / divide (平铺与切块)")

step("5.1 logical_product: 把 atom 平铺")
atom = Layout((8, 64), (64, 1))  # FA2 的 smem atom: 8 行 x 64 列
lp = logical_product(atom, Layout((16, 2)))  # 平铺 16 个 "行 tile" x 2 个 "列 tile"
print("atom =", atom)
print("logical_product(atom, (16,2)) =")
print(" ", lp)
print("读法: ((8,64),(16,2))  ——  mode0 = atom 内部, mode1 = 第几个 (行tile, 列tile)")
print("      每个 (行tile,列tile) 偏移 512/8192, 正好是 atom 的 cosize=512")

# tiled_product: 把两个 "平铺计数" 拍平成两个独立 mode
tp = tiled_product(atom, Layout((16, 2)))
print("tiled_product =", tp)

step("5.2 tile_to_shape (模仿 C++ cute::tile_to_shape)")
t = tile_to_shape(atom, (128, 128))
print("tile_to_shape(atom, (128,128)) =", t)
print("读法: 行 mode = (8,16):(64,512)  ->  atom 内 8 行 + 16 个 tile 行")
print("      列 mode = (64,2):(1,8192)  ->  atom 内 64 列 + 2 个 tile 列")


# 用暴力循环验证
def brute(R, C):
    return (R // 8 + (C // 64) * 16) * 512 + (R % 8) * 64 + (C % 64)


assert all(t(R, C) == brute(R, C) for R in range(128) for C in range(128))
print("暴力验证: tile_to_shape 与手写公式一致  OK")
print(
    "对比 5.1 的 logical_product: 平铺计数在 tile_to_shape 里是逐原 mode 的独立 mode "
    "((8,16),(64,2)),"
)
print(
    "logical_product 则嵌套成一个 mode ((8,64),(16,2)) —— 两者作为函数相同, "
    "只是 '形状视角' 不同。"
)

step("5.3 zipped_divide: 切块 (FA2 local_tile 的代数)")
# FA2 的 local_tile 用 Shape 形式的 tiler (二维 tile 坐标), 逐 mode 切:
A = Layout((200, 128), (128, 1))  # FA2 真实的 mQ: (seqlen=200, d=128) 行主序
zd = zipped_divide(A, (128, 128))
print("zipped_divide(mQ (200,128):(128,1), (128,128)) =")
print(" ", zd)
print("读法: mode0 = tile 内部 (128,128) 行主序;")
print(
    "      mode1 = tile 索引 (m_block, 0): 2 个行块 (d 横跨全宽, 所以第二维 size 1), "
    "每行块偏移 16384 = 128*128。"
)
print("注意: 第 2 个行块 (行 128..255) 越过了 seqlen=200 —— local_tile 照切,")
print("第 8 章的 M 边界谓词负责丢弃无效行。")
# 章头定义的 logical_divide 也有 1D 形式: "切出一块", 例如:
ld = logical_divide(Layout(24, 2), 8)
print(
    "logical_divide(24:2, 8) =",
    ld,
    "  # mode0 = 前 8 个元素 (步长随 A); mode1 = 有多少个这样的块 (步长 = 8*2)",
)
print(
    "by-mode 形式在第 9 章直接用: logical_divide(acc_v, (None, None, 2))  "
    "# 末 mode 切半, 其余不动"
)
print("对比: tiler 传 Layout 对象 (而非 shape 元组) 时, 是按 '连续 4096 个元素' 切:")
print(" ", zipped_divide(Layout((128, 128), (1, 128)), Layout((64, 64))))
print(" (列主序输入下, 每个 'tile' 其实是 128x32 的竖条, 不是 64x64 方块)")

step("5.4 FA2 代码对应 (kernel_traits.h)")
print("""
  // smem 布局 = swizzled atom 平铺到 (kBlockM, kHeadDim):
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{},
                                             Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  // gmem 大张量切出本 block 的 tile (flash_fwd_kernel.h 里的 local_tile):
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));
  // local_tile 本质就是 zipped_divide + 取第 m_block 个 tile。
""")

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 把 atom (8,32):(32,1) 平铺到 (64,96) (用 common.tile_to_shape),
    # 先手算预期的 shape/stride, 再打印并用暴力循环验证。
    atom = Layout((8, 32), (32, 1))  # 给定的 atom
    t = None  # <-- 填
    assert t is not None, "请填写 t"

    def brute(R, C):
        return (R // 8 + (C // 32) * 8) * 256 + (R % 8) * 32 + (C % 32)

    # 正确的 t 应满足: (a) 第一个 tile 与 atom 逐点一致;
    #                (b) 整体等于手写公式;  (c) shape/stride 与手算一致。
    first_tile_ok = all(t(r, c) == atom(r, c) for r in range(8) for c in range(32))
    ok = (
        first_tile_ok
        and all(t(R, C) == brute(R, C) for R in range(64) for C in range(96))
        and str(t) == "((8, 8), (32, 3)):((32, 256), (1, 2048))"
    )
    check(f"练习1: 平铺 (8,32) -> (64,96), 结果 {t}", ok)


def ex2():
    # 练习 2: zipped_divide(mQ (200,128):(128,1), (128,128)) 的结果是
    #   ((128,128), (2,1)):((128,1), (16384, 0))
    # 验证三件事: (a) tile 内部 size == 16384;
    #             (b) tile 索引第一维 (行块) 的 stride == 16384;
    #             (c) tile 索引第二维 (列块) 的 size == 1 (d 横跨全宽)。
    A = Layout((200, 128), (128, 1))  # 给定的 mQ
    d = None  # <-- 填
    assert d is not None, "请填写 d"
    ok = (
        size(d[0]) == 16384
        and d[1][0].stride == 16384
        and d[1][1].shape == 1
        and d[0].shape[1] == A.shape[1]
    )
    check(f"练习2: {A} 切块后的内部/tile 结构", ok)


def ex3():
    # 练习 3: 不用 tile_to_shape, 手写一个与
    #   tile_to_shape(Layout((8,64),(64,1)), (128,128))  "作为函数完全相同" 的
    # 嵌套 layout, 并用 assert_same 验证。
    # 提示: 行 mode = (atom 内 8 行, 多少个 tile 行),
    #       列 mode = (atom 内 64 列, 多少个 tile 列);
    #       tile 行 stride = 一个 atom 的 cosize (8 行 x 64 列);
    #       tile 列 stride = 一整列 atom (所有 tile 行 x 一个 atom 的 cosize)。
    hand = None  # <-- 填 Layout
    ref = tile_to_shape(Layout((8, 64), (64, 1)), (128, 128))
    assert hand is not None, "请填写 hand"
    assert_same(hand, ref, domain_2d(128, 128), "练习3")
    check("练习3: 手写 layout 与 tile_to_shape 函数相同", True)


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)

banner("第 5 章结束。下一章: Swizzle (smem bank conflict 的解法)")
