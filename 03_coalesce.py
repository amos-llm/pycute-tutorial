"""第 3 章: coalesce —— 把 layout 化简

目标: 理解 "同一个函数可以有多种写法"。coalesce 把嵌套 mode 合并成
最简形式, 合并规则一共 4 条 (见 csrc/cutlass/media/docs/cpp/cute/02_layout_algebra.md):

  对相邻两个 mode  s0:d0 ++ s1:d1:
    1. s0:d0  ++  _1:d1  =>  s0:d0          (大小为 1 的 mode 直接删)
    2. _1:d0  ++  s1:d1  =>  s1:d1          (同上)
    3. s0:d0  ++  s1:(s0*d0)  =>  s0*s1:d0  (恰好连续, 合并)
    4. 其他 => 保持不变

运行: python3 03_coalesce.py
"""

from common import banner, check, exercise, step, tile_to_shape
from pycute import coalesce, flatten, Layout, size

banner("第 3 章: coalesce —— 把 layout 化简")

step("3.1 基本化简")
L1 = Layout(((4, 2), 3), ((8, 2), 4))
print("L1 =", L1)
print("coalesce(L1) =", coalesce(L1))
print(
    "  # (4,2): 4*8=32 != 2, 不能合并; 后面的 2:2 和 3:4 恰好连续 (2*2=4), 合并成 6:2"
)

L2 = Layout((2, 1, 3), (1, 7, 2))
print("\nL2 =", L2)
print("coalesce(L2) =", coalesce(L2))
print("  # 中间的 1:7 被删掉; 2:1 与 3:2 连续 -> 6:1")

L3 = Layout((12, 12), (1, 12))
print("\nL3 =", L3, " (列主序方阵)")
print("coalesce(L3) =", coalesce(L3))

step("3.2 不能合并的例子")
L4 = Layout((4, 4), (1, 8))
print("\nL4 =", L4)
print("coalesce(L4) =", coalesce(L4), "  (4*1=4 != 8, 中间有空洞, 合并不了)")

step("3.3 FA2 里的例子: smem tile 布局的化简")
# kBlockM=128, kHeadDim=128, atom = 8 行 x 64 列 (kBlockKSmem=64)
# (tile_to_shape 是 common.py 的 helper, 第 5 章正式讲, 此处只看 coalesce 效果)
atom = Layout((8, 64), (64, 1))
tiled = tile_to_shape(atom, (128, 128))
print("atom =", atom)
print("tile_to_shape(atom, (128,128)) =", tiled)
print("coalesce 后:", coalesce(tiled))
print(
    "  # (8,16):(64,512) 恰好连续 -> (128):(64); (64,2) 中间隔了整整一个 atom, 合并不了"
)

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 先手算 coalesce(((4,2),3):((2,8),16)) 的结果, 填成 Layout 验证。
    # 提示: 展开成 4:2, 2:8, 3:16 逐个看规则 3 是否成立。
    your_answer = None  # <-- 填 Layout
    assert your_answer is not None, "请填写 your_answer"
    L = Layout(((4, 2), 3), ((2, 8), 16))
    ok = your_answer == coalesce(L) and all(
        your_answer(c) == L(c) for c in domain_1d(size(L))
    )
    check(f"练习1: coalesce 结果 (你的: {your_answer})", ok)


def ex2():
    # 练习 2: 写一个含 size-1 mode 的 layout, 使 coalesce 后变成 8:1。
    your_layout = None  # <-- 填 Layout
    assert your_layout is not None, "请填写 your_layout"
    ok = coalesce(your_layout) == Layout(8, 1) and any(
        s == 1 for s in flatten(your_layout.shape)
    )
    check(f"练习2: coalesce({your_layout}) == 8:1", ok)


def ex3():
    # 练习 3 (思考, 无自动判分): 为什么 coalesce 不能把 (4,4):(1,8) 合并?
    # 如果强行合并成 16:1, 同一 (行,列) 位置的偏移会变。对比下面两列:
    A = Layout((4, 4), (1, 8))
    B = Layout(16, 1)
    for c in range(4):
        print(
            f"  位置 ({c},1): 原布局偏移 {A(c, 1)}  vs  强行合并 16:1 后 {B(c * 4 + 1)}"
        )
    print("  (结论: 只要某个 mode 的 stride != 前一个 mode 的 size*stride, 就不能合并)")


# 小工具: 一维 domain
def domain_1d(n):
    return range(n)


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3(思考)", ex3)

banner("第 3 章结束。下一章: composition (核心操作)")
