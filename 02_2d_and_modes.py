"""第 2 章: 二维布局与嵌套 mode

目标: 看懂 (Shape, Stride) 的任意嵌套形式, 比如 FA2 里常见的
    Layout<Shape<Shape<_4,_2>, _2>, Stride<Stride<_2,_16>, _8>>
本质就是 "一个 mode 可以是一个 tuple, 坐标也可以嵌套"。

运行: python3 02_2d_and_modes.py
"""

from common import banner, check, exercise, show_grid, step
from pycute import cosize, crd2idx, flatten, idx2crd, Layout

banner("第 2 章: 二维布局与嵌套 mode")

step("2.1 行主序 vs 列主序: 只是 stride 不同")
rm = Layout((4, 8), (8, 1))  # 行主序: 一行连续
cm = Layout((4, 8), (1, 4))  # 列主序: 一列连续
print("行主序 (4,8):(8,1):")
show_grid(rm)
print("列主序 (4,8):(1,4):")
show_grid(cm)
print("注意: 两者是同一个 32 个元素的排列, 只是坐标->偏移的函数不同。")

step("2.2 跨步访问: stride 可以 > 1 (比如隔列取样)")
strided = Layout((4, 4), (8, 2))  # 从 8 列宽的大数组里, 隔列取 4 列, 行距 8
print("strided (4,4):(8,2): 每行取 4 列(步长 2), 行距 8:")
show_grid(strided, "从一个大数组里隔列读 4x4:")
print("cosize =", cosize(strided), "  (偏移范围 [0, 30])")

step("2.3 嵌套 mode: 一个 mode 本身是 tuple")
# ((2,2),4):((4,1),2): 第一个 mode 内部还有两个子坐标
N = Layout(((2, 2), 4), ((4, 1), 2))
print("N =", N)
print("N 的 rank =", len(N))
print("第 0 个 mode:", N[0], "  第 1 个 mode:", N[1])
print("flatten(shape) =", flatten(N.shape))
print("flatten(stride)=", flatten(N.stride))
print("N((1,0),3) =", N((1, 0), 3), "  # 1*4 + 0*1 + 3*2 = 10")
print("N((0,1),2) =", N((0, 1), 2), "  # 0*4 + 1*1 + 2*2 = 5")

step("2.4 用 None 做切片 (等价于 C++ 的 _)")
L = Layout((4, 8), (8, 1))  # 行主序 4x8
print("L(2, None)  =", L(2, None), "  # 第 2 行: 一个 1D 子 layout")
print("L(None, 3)  =", L(None, 3), "  # 第 3 列: 步长 8")
print("第 2 行起始偏移 = crd2idx((2,0),(4,8),(8,1)) =", crd2idx((2, 0), (4, 8), (8, 1)))

step("2.5 FA2 中真实出现的嵌套 layout (kernel_traits.h)")
print("""
  using GmemLayoutAtom =
      Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
             Stride<Int<kGmemThreadsPerRow>, _1>>;
  // 取 kNThreads=128, kGmemThreadsPerRow=8:
""")
G = Layout((16, 8), (8, 1))  # 线程 t -> (row, col_block)
print("GmemLayoutAtom (16,8):(8,1):  thread t 的坐标 = idx2crd(t, (16,8), (8,1))")
for t in (0, 5, 10, 127):
    print(
        f"  t={t:>3} -> (row={t // 8}, colblock={t % 8})   "
        f"[pycute: {idx2crd(t, (16, 8), (8, 1))}]"
    )

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 用 Layout 表达 "6x8 矩阵, 元素 (r,c) 的偏移 = 2*r + c"
    # (相当于一个大数组里, 行距为 2 的 6 行 x 8 列子块)
    your_L = None  # <-- 填 Layout
    assert your_L is not None, "请填写 your_L"
    ok = all(your_L(r, c) == 2 * r + c for r in range(6) for c in range(8))
    check("练习1: (r,c) -> 2r+c", ok)


def ex2():
    # 练习 2: N = ((2,2),4):((4,1),2), 求 N((1,1),1) 的值, 并验证。
    N = Layout(((2, 2), 4), ((4, 1), 2))
    your_answer = None  # <-- 填数字
    assert your_answer is not None, "请填写 your_answer"
    check("练习2: N((1,1),1)", your_answer == N((1, 1), 1))


def ex3():
    # 练习 3: 对行主序 (4,8):(8,1), 求 "第 3 行第 2 列" 的偏移,
    # 并用两种方法各算一次, 确认一致: (a) 直接调用 L(r,c); (b) crd2idx。
    L = Layout((4, 8), (8, 1))
    via_call = L(3, 2)
    via_crd2idx = crd2idx((3, 2), (4, 8), (8, 1))
    check("练习3: 两种方法一致", via_call == via_crd2idx == 26)


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)

banner("第 2 章结束。下一章: coalesce (化简)")
