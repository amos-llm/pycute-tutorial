"""第 9 章: MMA 寄存器碎片与 "寄存器代数"

目标: 理解 FA2 最精妙的一招 (utils.h 的 convert_layout_acc_Aregs):
把第一个 GEMM 的累加器 (P, fp32) **不搬一个寄存器**, 直接重新解释成
第二个 GEMM 的 A 碎片 (P, fp16)。

背景: mma.sync.m16n8k16 (fp16):
  A 操作数 (16x16): 每线程 8 个 fp16 (4 个 .b32 寄存器)
  C 操作数 (16x8) : 每线程 4 个 fp32

本章 9.3 节用 FA2 hdim128 (sm8x) 的真实编译配置:
  kBlockM=128, kBlockN=32, kNWarps=4 (128 线程)。
  S 瓦片 (128,32): acc_s 每线程 (MMA=4, MMA_M=2, MMA_N=4) = 32 个 fp32。

运行: python3 09_mma_fragments.py
"""

from pycute import Layout, idx2crd, logical_divide, make_layout, size

from common import banner, check, exercise, step

banner("第 9 章: MMA 寄存器碎片与 '寄存器代数'")

step("9.1 m16n8k16 的 A 碎片: 每线程拿 8 个元素")
print("""
  线程 t: group = t//4 (0..7), tig = t%4 (0..3)
  8 个值 (fp16) 位于 (16x16) A tile 的:
    v0: (row=group,      col=tig*2 + 0)
    v1: (row=group,      col=tig*2 + 1)
    v2: (row=group + 8,  col=tig*2 + 0)
    v3: (row=group + 8,  col=tig*2 + 1)
    v4: (row=group,      col=tig*2 + 0 + 8)
    v5: (row=group,      col=tig*2 + 1 + 8)
    v6: (row=group + 8,  col=tig*2 + 0 + 8)
    v7: (row=group + 8,  col=tig*2 + 1 + 8)
  写成 TV 布局 (A tile 行主序 (16,16):(16,1)):
    T = (4,8):(2,16)      # (tig, group)
    V = (2,2,2):(1,128,8) # (列内pair, 行半区, k半区)   v = cp + 2*rh + 4*kh
""")
T_A = Layout((4, 8), (2, 16))
V_A = Layout((2, 2, 2), (1, 128, 8))


def a_off(t, v):
    """线程 t 的第 v 个值在 (16,16) tile (行主序) 里的偏移。"""
    return T_A(t % 4, t // 4) + V_A(*idx2crd(v, (2, 2, 2)))


# 验证: 32 线程 x 8 值 = 256 = 16*16, 且不重不漏
seen = set()
for t in range(32):
    for v in range(8):
        o = a_off(t, v)
        assert 0 <= o < 256 and o not in seen, f"冲突! t={t}, v={v}, off={o}"
        seen.add(o)
print("A 碎片覆盖 (16,16) 恰好一次: OK")


# 再逐项对照上面的 PTX 表 (能抓住 "步长写反" 这类覆盖检查发现不了的错误):
def ptx_a16(t, v):
    cp, rh, kh = idx2crd(v, (2, 2, 2))  # v = cp + 2*rh + 4*kh
    return ((t // 4) + 8 * rh) * 16 + ((t % 4) * 2 + cp + 8 * kh)


assert all(a_off(t, v) == ptx_a16(t, v) for t in range(32) for v in range(8))
print("a_off 与 PTX 表逐项一致: OK")

step("9.2 m16n8 的 C 碎片: 每线程拿 4 个 fp32")
print(
    """
  C tile (16x8) 行主序 (16,8):(8,1):
    v0: (row=group,      col=tig*2 + 0)
    v1: (row=group,      col=tig*2 + 1)
    v2: (row=group + 8,  col=tig*2 + 0)
    v3: (row=group + 8,  col=tig*2 + 1)
  注意: 行距是 8 不是 16! 所以 T_C = (4,8):(2,8) (与 T_A 不同)
  T_C = (4,8):(2,8)   (V_C 在练习 2 让你自己写: 照上表手算; 下面代码里"""
    """只是 PTX 一致性检查用)
"""
)
T_C = Layout((4, 8), (2, 8))
V_C = Layout((2, 2), (1, 64))


def c_off(t, v):
    return T_C(t % 4, t // 4) + V_C(v % 2, v // 2)


def ptx_c(t, v):
    g, tig = t // 4, t % 4
    row, col = g + 8 * (v // 2), tig * 2 + (v % 2)
    return row * 8 + col


assert all(c_off(t, v) == ptx_c(t, v) for t in range(32) for v in range(4))
print("C 碎片与 PTX 规则一致: OK")

step("9.3 FA2 的一招: acc_s (P, fp32) -> 第二个 GEMM 的 A 碎片")
print(
    """
  真实配置: kBlockM=128, kBlockN=32, kNWarps=4 (sm8x hdim128 标准配置)。
  第一个 GEMM: S = Q @ K^T, acc_s 形状 (MMA=4, MMA_M=2, MMA_N=4):
    MMA_M = 128 行 / (16行/瓦片 x 4 warp) = 2,   MMA_N = 32 列 / 8列 = 4。
  每线程 4*2*4 = 32 个 fp32, 128 线程 x 32 = 4096 = 128*32 ✓ 不重不漏。

  第二个 GEMM: O = P @ V, P 是 A 操作数 (M=128, K=kBlockN=32)。
  它的 A 碎片把 K=32 看成 2 个 16 块; 每个 16 块 = 两个相邻的 C tile (N=8)。
  对应关系: nn = 2*kb + kh (kb = 第几个 16 块, kh = 块内 k 半区), 即 """
    """nn//2 是 k 块号, nn%2 是半区。
  关键巧合: C tile 的 4 个值 (c0,c1,c2,c3) 的顺序
      = (row, col), (row, col+1), (row+8, col), (row+8, col+1)
    与 A tile 前 4 个值 (a0..a3) 的顺序完全一致!
  所以只需把值索引 (a, mm, nn) 重新分组为 ((a, nn%2), mm, nn//2),
  即 FA2 的 ((4,2), MMA_M, MMA_N/2)。没有任何数据搬运。
"""
)
# 每线程 acc 值布局, S 瓦片 (128,32) 行主序 (行距 32):
#   a%2: 列+1 -> 1;  a//2: 行+8 -> 8*32 = 256;  mm: 行+64 -> 2048;  nn: 列+8 -> 8
# 教学说明: 这里把每线程的 acc 值用 "S 瓦片 (128,32) 行主序" 里的位置来写,
# 便于和 9.3 的 PTX 对照; cute 真实的 FrgTypeC 是紧凑布局,
# 但 mode 结构相同 ((2,2), MMA_M, MMA_N)。
acc_v = Layout(((2, 2), 2, 4), ((1, 256), 2048, 8))
print(f"acc 值布局 (a,mm,nn): {acc_v}  size = {size(acc_v)}")

# FA2 代码 (utils.h): logical_divide(acc, (_,_,_2)) 再重排
div = logical_divide(acc_v, (None, None, 2))
print(f"logical_divide(acc, (None,None,2)) = {div}")
regrouped = make_layout(make_layout(div[0], div[2][0]), div[1], div[2][1])
print(f"重排后 ((4,2), mm, nn/2) = {regrouped}")


# 验证: 重排后第 v 个值 (v = cp + 2*rh + 4*kh, 即 PTX A 碎片的寄存器序)
# 必须正好落在 mma 期望的 (row, k) 上。128 线程: w = t//32, g = (t%32)//4, tig = t%4。
def ptx_afrag_reg(t, v, mm, kb):
    """第 (mm, kb) 个 mma 步, 寄存器 v 的 (row,k)
    在 (128,32) 瓦片 (行距 32) 里的偏移。"""
    cp, rh, kh = idx2crd(v, (2, 2, 2))
    row = 64 * mm + 16 * (t // 32) + (t % 32) // 4 + 8 * rh
    k = 16 * kb + (t % 4) * 2 + cp + 8 * kh
    return row * 32 + k


def base(t):
    return (16 * (t // 32) + (t % 32) // 4) * 32 + (t % 4) * 2


ok = all(
    regrouped(((a % 2, a // 2), nn % 2), mm, nn // 2)
    == ptx_afrag_reg(t, (a % 2) + 2 * (a // 2) + 4 * (nn % 2), mm, nn // 2) - base(t)
    for t in range(128)
    for a in range(4)
    for mm in range(2)
    for nn in range(4)
)
print("重排后的寄存器顺序 == PTX A 碎片顺序: OK" if ok else "不匹配!")

step("9.4 FA2 代码对应")
print(
    """
  utils.h convert_layout_acc_Aregs (m16n8k16 分支, mma_shape_K==16):
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // MMA_N -> (2, MMA_N/2)
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    // 即 (4, MMA_M, MMA_N) -> ((4,2), MMA_M, MMA_N/2)

  flash_fwd_kernel.h:
    Tensor rP = convert_type<Element>(acc_s);                    """
    """// fp32 -> fp16 (仍是 acc 顺序)
    Tensor tOrP = make_tensor(rP.data(), """
    """convert_layout_acc_Aregs<TiledMma>(rP.layout()));
    gemm_rs(acc_o, tOrP, tOrVt, tOsVt, ...);                     """
    """// 直接喂给第二个 mma
"""
)

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 验证 A 碎片 (T_A, V_A) 覆盖 (16,16) 恰好一次。
    # 写循环收集所有 (t,v) 的偏移, 断言: 共 256 个, 无重复, 范围 [0,256)。
    seen = set()
    ok = True
    for t in range(32):
        for v in range(8):
            g, tig = t // 4, t % 4
            vv = idx2crd(v, (2, 2, 2))  # (列pair, 行半区, k半区)  与 9.1 的 V_A 一致
            row = g + 8 * vv[1]
            col = tig * 2 + vv[0] + 8 * vv[2]
            o = row * 16 + col
            if o in seen or not (0 <= o < 256):
                ok = False
            seen.add(o)
    check("练习1: A 碎片覆盖 (16,16) 恰好一次", ok and len(seen) == 256)


def ex2():
    # 练习 2: 写出 C 碎片 (16x8) 的值布局 V_C, 验证 v0..v3 与 PTX 规则一致。
    # 提示: 用 9.2 节的 T_C (C tile 行距 8, 不是 T_A 的 16)。
    your_V_C = None  # <-- 填 Layout, 如 (2,2):(1,?)
    assert your_V_C is not None, "请填写 your_V_C"
    ok = all(
        T_C(t % 4, t // 4) + your_V_C(v % 2, v // 2) == ptx_c(t, v)
        for t in range(32)
        for v in range(4)
    )
    check(f"练习2: C 碎片值布局 {your_V_C}", ok)


def ex3():
    # 练习 3 (综合): 把 9.3 节的重排流程补完整 —— 给定 acc_v,
    # 用 logical_divide(acc_v, (None, None, 2)) + make_layout 得到重排 layout,
    # 并验证它和 "直接按 PTX A 碎片规则" 的 ptx_afrag_reg 一致 (参照 9.3 节的验证代码)。
    div2 = None  # <-- 填 logical_divide(acc_v, (None, None, 2))
    assert div2 is not None, "请填写 div2"
    r = make_layout(make_layout(div2[0], div2[2][0]), div2[1], div2[2][1])
    ok = all(
        r(((a % 2, a // 2), nn % 2), mm, nn // 2)
        == ptx_afrag_reg(t, (a % 2) + 2 * (a // 2) + 4 * (nn % 2), mm, nn // 2)
        - base(t)
        for t in range(128)
        for a in range(4)
        for mm in range(2)
        for nn in range(4)
    )
    check("练习3: 完整重排流程", ok)


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)

banner("第 9 章结束。最后一章: FA2 端到端 —— 追一个元素")
