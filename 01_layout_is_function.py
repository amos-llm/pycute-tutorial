"""第 1 章: Layout 就是函数

目标: 建立核心心智模型 ——
    Layout = (Shape, Stride) = 一个从 "坐标" 到 "线性偏移" 的函数。

FA2 对应代码: csrc/flash_attn/src/kernel_traits.h 里所有
    using XXX = Layout<Shape<...>, Stride<...>>
都是这种函数的声明。

运行: python3 01_layout_is_function.py
"""

from pycute import Layout, cosize, crd2idx, idx2crd, size

from common import banner, check, exercise, show_grid, step

banner("第 1 章: Layout 就是函数")

step("1.1 最简单的 layout: 一维")
L = Layout(64)  # Shape=64, 不写 stride 时默认 "列主序" (一维就是连续)
print(f"L = {L}")  # 64:1   读作: 形状 64, 步长 1
print(f"L(7)  = {L(7)}")  # 坐标 7 -> 偏移 7*1 = 7
print(f"size  = {size(L)}")  # 定义域大小 (输入坐标的个数)
print(f"cosize= {cosize(L)}")  # 值域跨度 (输出偏移覆盖的范围)

L2 = Layout(64, 2)  # 步长为 2: 取每隔一个元素
print(f"\nL2 = {L2}")
print(f"L2(7) = {L2(7)}")  # 7*2 = 14
print(f"cosize = {cosize(L2)}")  # L2(63) = 126, 所以 cosize = 127

step("1.2 二维: offset = coord0*stride0 + coord1*stride1")
# (8,4):(4,1): 8 行 4 列, 行主序 (最后一个 mode 步长为 1)
A = Layout((8, 4), (4, 1))
print(f"A = {A}")
print(f"A(3,2) = {A(3, 2)}  # 3*4 + 2*1 = 14")
show_grid(A, "A 的偏移矩阵 (每行一个 r, 每列一个 c):")

step("1.3 逆映射: 偏移 -> 坐标 (idx2crd / crd2idx)")
# crd2idx(crd, shape, stride=None): None 表示列主序
print(
    f"\ncrd2idx((3,2), (8,4))          = {crd2idx((3, 2), (8, 4))}  # 列主序: 3 + 2*8 = 19"
)
print(
    f"crd2idx((3,2), (8,4), (4,1))    = {crd2idx((3, 2), (8, 4), (4, 1))}  # 行主序: 3*4 + 2 = 14"
)
print(
    f"idx2crd(19, (8,4))              = {idx2crd(19, (8, 4))}  # 19 = 3 + 2*8 -> (3,2)"
)

step("1.4 默认方向: pycute 和 C++ cute 都是列主序 (重要!)")
print("""
  pycute  : Layout((8,4))          默认列主序 -> (8,4):(1,8)   [本教程环境]
  C++ cute: make_layout(shape)     默认 stride = compact_major<LayoutLeft>(shape)
                                    LayoutLeft = 列主序 (cute/stride.hpp)
                                    -> make_layout((8,4)) = (8,4):(1,8), 和 pycute 一致!
  行主序必须显式指定: compact_row_major / 自定义 stride 生成器。
  FA2 代码里大量出现 GenRowMajor (cute 里还有 GenColMajor) 这种
  显式声明 stride 方向的"生成器"类型。
""")
print(f"pycute: Layout((8,4))      = {Layout((8, 4))}")
print(f"pycute: Layout((8,4),(4,1))= {Layout((8, 4), (4, 1))}  # 显式行主序")

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 手工算出 Layout((4,3),(3,1))(2,1) 的值, 填入下面并断言相等。
    # 提示: (i,j) -> i*3 + j*1
    your_answer = None  # <-- 填数字
    assert your_answer is not None, "请填写 your_answer"
    L = Layout((4, 3), (3, 1))
    check("练习1: (4,3):(3,1) 在 (2,1) 的偏移", your_answer == L(2, 1))


def ex2():
    # 练习 2: 用 idx2crd 求 layout (4,3):(1,4) (列主序: 第一个 mode 步长为 1)
    # 中偏移 7 对应的坐标,
    # 填成 (i, j) 元组。
    your_crd = None  # <-- 填 (i, j)
    assert your_crd is not None, "请填写 your_crd"
    got = idx2crd(7, (4, 3), (1, 4))
    check("练习2: (4,3):(1,4) 中偏移 7 的坐标", your_crd == got)


def ex3():
    # 练习 3: 构造一个 size != cosize 的 layout (即不是连续的), 并验证。
    # 提示: 步长之间留空隙, 比如 (8,4):(2,1)
    your_layout = None  # <-- 填一个 Layout
    assert your_layout is not None, "请填写 your_layout"
    ok = size(your_layout) != cosize(your_layout)
    check(
        f"练习3: 构造 size!=cosize (你的: {your_layout}, size={size(your_layout)}, cosize={cosize(your_layout)})",
        ok,
    )


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)

banner("第 1 章结束。下一章: 二维与嵌套 mode")
