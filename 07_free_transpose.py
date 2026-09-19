"""第 7 章: 免费的转置 —— FA2 的 SmemLayoutVtransposed

目标: 看懂 FA2 里 "不搬数据就能换视角" 的代数技巧:

  using SmemLayoutVtransposed =
      composition(SmemLayoutKV{},
                  make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{}));

背景:
  FA2 的两个 GEMM:
    S = Q K^T      : A=Q (m,k),  B=K^T (n,k)
    O = P V        : A=P (m,k=n), B=V^T (d, n)     <- 需要把 V "看成" (d, n)
  smem 里 V 是按 (n, d) 存的 (和 K 一样的布局, 方便 cp.async 进来)。
  但第二个 GEMM 需要 V^T = (d, n)。

运行: python3 07_free_transpose.py
"""

from common import (
    assert_same,
    banner,
    check,
    domain_2d,
    exercise,
    step,
    SwizzledLayout,
)
from pycute import coalesce, composition, cosize, Layout, Swizzle

banner("第 7 章: 免费的转置 —— FA2 的 SmemLayoutVtransposed")

step("7.1 问题: 同一块 smem, 两个视角")
n, d = 64, 128
sV = Layout((n, d), (d, 1))  # (n, d): d 连续 (FA2 的 SmemLayoutKV 的非 swizzle 部分)
print("sV (n,d) =", sV, "   sV(n,d) = n*128 + d")
print("但 O = P@V 的 mma 需要 B 矩阵是 (d, n): 元素 (i=d, j=n) 应该读到 sV(j, i)。")

step("7.2 代数解法: composition(sV, 行主序(d,n))")
# B = (d, n) 行主序读法: (i,j) -> i*n + j  (把 (d,n) 坐标先变成 1D 索引)
B = Layout((d, n), (n, 1))
Vt = composition(sV, B)
print("B  (d,n) row-major =", B)
print("Vt = sV o B =", Vt, "  (d,n)")
print("验证 Vt(i,j) == sV(j,i):")
assert_same(Vt, lambda i, j: sV(j, i), domain_2d(d, n), "Vt")
print("  全部相等, OK")
print("注意: Vt 的 cosize 和 sV 相同, 没有任何额外存储 —— 这就是 '免费转置'。")

step("7.3 为什么转置后的 1D 函数是恒等 (coalesce 后)?")
print("coalesce(Vt) =", coalesce(Vt))
print("# Vt = (128,64):(1,128): 两个 mode 恰好连续 (128*1 = 128 = 第二 mode 步长),")
print("# 所以 coalesce 后是 8192:1 —— 1D 化就是恒等函数。")
print("# 结构 (128,64):(1,128) 才是重点: 它告诉 mma '第 i 行第 j 列在哪个偏移'。")

step("7.4 带上 swizzle 依然成立: 转置与 swizzle 可交换")
swz = Swizzle(3, 3, 3)
sV_swz = SwizzledLayout(swz, sV)  # 构造 1: 先 swizzle 再转置: sV_swz(j, i)
# 构造 2: 先转置 (7.2 的 Vt) 再 swizzle: swz(Vt(i, j))
ok = all(swz(Vt(i, j)) == sV_swz(j, i) for i in range(d) for j in range(n))
print("两种独立构造一致: swz(Vt(i,j)) == sV_swz(j,i), 对所有 (i,j) 成立:", ok)
print(
    "# 即 '先转置后 swizzle' == '先 swizzle 后转置': swizzle 只作用于最终偏移, "
    "不影响 '换视角' 的代数。"
)

step("7.5 FA2 代码对应 (kernel_traits.h / flash_fwd_kernel.h)")
print(
    """
  kernel_traits.h:
    using SmemLayoutVtransposed =
        composition(SmemLayoutKV{},
                    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{}));
    using SmemLayoutVtransposedNoSwizzle =
        get_nonswizzle_portion(SmemLayoutVtransposed{});   // 把 swizzle 拆出来

  flash_fwd_kernel.h:
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  """
    """// 碎片结构 (寄存器布局) 用无 swizzle 视图算
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);           """
    """// smem 拷贝地址用 swizzled 视图算
"""
)

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: hdim96 的 K (n=64, d=96) 按 (n,d) 存: sK = (64,96):(96,1)。
    # 写出它的转置视图 (d,n) 的 composition 表达式, 并验证 == sK(j,i)。
    sK = Layout((64, 96), (96, 1))
    Kt = None  # <-- 填 composition(sK, ...)
    assert Kt is not None, "请填写 Kt"
    assert_same(Kt, lambda i, j: sK(j, i), domain_2d(96, 64), "Kt")
    check(f"练习1: K^T 视图 = {Kt}", True)


def ex2():
    # 练习 2 (通用化): M=8, N=5, s = (M,N):(N,1) 行主序。
    # 用 composition 表达它的转置 (N,M):(1,M), 并验证。
    M, N = 8, 5
    s = Layout((M, N), (N, 1))
    st = None  # <-- 填 composition(s, ...)
    assert st is not None, "请填写 st"
    assert_same(st, lambda i, j: s(j, i), domain_2d(N, M), "st")
    check(f"练习2: 通用转置 {M}x{N} -> {st}", True)


def ex3():
    # 练习 3 (思考, 打印你的答案): 转置为什么 "免费"?
    # 1) 转置视图和原布局的 cosize 相同吗? (打印验证)
    s = Layout((4, 6), (6, 1))
    st = composition(s, Layout((6, 4), (4, 1)))
    print("cosize(s) =", cosize(s), " cosize(st) =", cosize(st))
    # 2) 它为什么不需要额外内存? (一句话)
    print("你的回答: <在这里写一句话>")


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3(思考)", ex3)

banner("第 7 章结束。下一章: identity layout 与谓词")
