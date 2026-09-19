"""第 8 章: identity layout 与谓词 (边界检查的代数)

目标: 看懂 FA2 里这两处代码是怎么工作的:

    // flash_fwd_kernel.h:223,239 — 构造 identity tensor 并做和真实数据相同的切分
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);   // 每值带上 (行, 列) 坐标
    tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;        // 我的第 k 个值在全局第几列?

    // utils.h copy() 里, M 方向 (行) 的边界:
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) { ... }

核心思想: 对 "恒等映射" 做和真实数据完全相同的切分,
每个值就自动带上了 "自己是全局第 (行, 列)" 的信息, 谓词直接查坐标。

本章配置取 FA2 hdim128 非 causal 的标准编译配置
(flash_fwd_launch_template.h: run_mha_fwd_hdim128):
    kBlockM=128, kHeadDim=128, kNWarps=4 (128 线程), kBlockKSmem=64。

运行: python3 08_identity_and_predicates.py
"""

from common import banner, check, exercise, step
from pycute import idx2crd, Layout

banner("第 8 章: identity layout 与谓词")

step("8.1 identity tensor = 坐标即数据的 layout")
M, K = 128, 128
id_t = Layout((M, K), (K, 1))  # (m,k) -> m*K + k: "偏移" 本身就是坐标
print("id_t =", id_t)
print("id_t(5, 20) =", id_t(5, 20))
print(
    "还原坐标: idx2crd(660, (128,128), (128,1)) =", idx2crd(id_t(5, 20), (M, K), (K, 1))
)
print("  # 注意要传 stride=(128,1) 行主序; 不传则按列主序, 会得到 (20,5) —— 又踩坑")

step("8.2 FA2 的 gmem->smem 拷贝布局 (hdim128 标准配置, 128 线程)")
print("""
  GmemLayoutAtom = (16,8):(8,1)   128 线程 -> (行块内行, 列块)   [kNThreads/8 x 8]
  value layout   = (8):(1)        每线程连读 8 个 k
  整个 (128,128) Q tile = 8 个行块(每块16行) x 2 个 k 页(每页64列)
""")
atom_thr = Layout((16, 8), (8, 1))
print(
    "线程 10 的 (行块内行, 列块) =",
    idx2crd(10, (16, 8), (8, 1)),
    " -> 行块内行 10//8=1, 列块 10%8=2 (k0=16)",
)


def thread_value_coord(t, mt, kp, v):
    """线程 t 在 (行块 mt, k 页 kp) 的第 v 个值, 在 (128,128) tile 里的
    (行, 列) 坐标。"""
    return mt * 16 + t // 8, kp * 64 + (t % 8) * 8 + v


step("8.3 对 identity 做同样的切分 -> 每个值带上 (行, 列) 坐标")
t = 10
print(f"线程 {t} 的 8 个值 (行块0, k页0) 的 tile 内坐标:")
for v in range(8):
    m, k = thread_value_coord(t, 0, 0, v)
    print(f"  v={v}: (m={m}, k={k})")
print("FA2 的 tQpQ(k) = get<1>(tQcQ(0,0,k)) < params.d  就是取这些坐标的 k 分量。")

step("8.4 谓词实战 A: M 边界 (最常见, seqlen_q 不是 128 的倍数)")
print("""
  场景: seqlen_q=200, 本 CTA 负责 m_block=1 (全局行 128..255)。
  有效行只有 128..199, 即 tile 内行 < 200-128 = 72。
  FA2 (utils.h copy(); max_MN 是调用点 flash_fwd_kernel.h:257 传入的实参):
      max_MN = binfo.actual_seqlen_q - m_block * kBlockM;   // = 72
      if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) { ... }
  线程 t 的 mt 个行块对应 tile 行 = mt*16 + t//8 (mt=0..7), 谓词 = 该行 < 72。
""")
seqlen_q, m_block, kBlockM = 200, 1, 128
max_MN = seqlen_q - m_block * kBlockM  # 72
print(f"max_MN = {max_MN}")
t = 10
preds = [thread_value_coord(t, mt, 0, 0)[0] < max_MN for mt in range(8)]
print(
    f"线程 {t} 的 8 个行块 "
    f"(行 = {[thread_value_coord(t, mt, 0, 0)[0] for mt in range(8)]}):"
)
print("  谓词:", preds)

cnt = 0
for t in range(128):
    for mt in range(8):  # 8 个 16 行块
        if thread_value_coord(t, mt, 0, 0)[0] < max_MN:
            cnt += 16  # 每 (t,mt) 写 16 个 k 值 (2 页 x 8)
print(f"有效元素总数 = {cnt}   (应为 72 行 x 128 k = {72 * 128})")

step("8.5 谓词实战 B: K 边界 (d 不是 64 的倍数, 被 padding 到 kHeadDim)")
print("""
  场景: 真实 head dim d=48, 不是 64 的倍数。
  C++ 侧 (flash_api.cpp) 先算 d_rounded = round_multiple(d, 32) = 64,
  再按真实 d 分桶 (HEADDIM_SWITCH: d≤32→hdim32, ≤64→hdim64, ≤96→hdim96, ...)
  选到 hdim64 kernel (kHeadDim=64)。params.d 仍是 48, 所以
  is_even_K = (params.d == kHeadDim) = (48 == 64) = false。
  hdim64 配置: kBlockKSmem=64, 1 个 k 页; GmemLayoutAtom 仍是 (16,8):(8,1)。
  线程 t 的 k 值 = (t%8)*8 + v (v=0..7), 谓词 k < 48:
      t%8 = 0..5 的值全有效, t%8 = 6,7 的值全无效。
  (FA2 要求 d 是 8 的倍数且 ≤256; kHeadDim ∈ {32,64,96,128,192,256},
   d≤128 时 d_rounded 取 32 的倍数, d>128 时取 64 的倍数;
   只要 d != kHeadDim 就需要这个谓词。若编译时定义了
   FLASHATTENTION_DISABLE_UNEVEN_K, 则断言 d == d_rounded, 即只支持
   d 恰好等于 kHeadDim 的情况。)
""")
d, kHeadDim = 48, 64
t = 54
ks = [(t % 8) * 8 + v for v in range(8)]
print(f"线程 {t} (t%8={t % 8}) 的 8 个 k 值: {ks}")
print("  谓词 k < 48:", [k < d for k in ks])
cnt = 0
for t in range(128):
    for mt in range(8):
        for v in range(8):
            if (t % 8) * 8 + v < d:
                cnt += 1
print(f"有效元素总数 = {cnt}   (应为 128 行 x 48 k = {128 * d})")

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 写出 128 线程映射到 (行块内行, 列块) 的 GmemLayoutAtom
    # (每行块 16 行, 每行 8 个列块), 并验证 t=42 -> (5, 2)。
    your_atom = None  # <-- 填 Layout
    assert your_atom is not None, "请填写 your_atom"
    ok = your_atom == Layout((16, 8), (8, 1)) and idx2crd(
        42, your_atom.shape, your_atom.stride
    ) == (5, 2)
    check("练习1: GmemLayoutAtom (hdim128)", ok)


def ex2():
    # 练习 2: 线程 t=10 (行块内行=1), seqlen_q=200, m_block=1 (max_MN=72)。
    # 它的 8 个行块对应 tile 行 1,17,33,49,65,81,97,113, 填出谓词列表。
    expected = [r < 72 for r in (1, 17, 33, 49, 65, 81, 97, 113)]
    your_pred = None  # <-- 填 8 个 bool
    assert your_pred is not None, "请填写 your_pred"
    check(f"练习2: your_pred={your_pred}", your_pred == expected)


def ex3():
    # 练习 3: 写循环统计 seqlen_q=200, m_block=1 时 Q tile 的有效元素总数,
    # 验证 == 72*128。提示: 对每个 (t, mt) 检查行 mt*16 + t//8 < 72,
    # 每个满足的 (t,mt) 贡献 16 个 k 值。
    cnt = 0
    for t in range(128):
        for mt in range(8):
            if mt * 16 + t // 8 < 72:
                cnt += 16
    check(f"练习3: 有效元素 {cnt} == 72*128", cnt == 72 * 128)


def ex4():
    # 练习 4 (K 边界): d=48, kHeadDim=64 (1 个 k 页), 128 线程 (16,8):(8,1)。
    # 线程 t 的 k = (t%8)*8 + v。统计有效元素总数, 验证 == 128*48。
    cnt = 0
    for t in range(128):
        for mt in range(8):
            for v in range(8):
                if (t % 8) * 8 + v < 48:
                    cnt += 1
    check(f"练习4: K 边界有效元素 {cnt} == 128*48", cnt == 128 * 48)


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)
exercise("练习4", ex4)

banner("第 8 章结束。下一章: MMA 寄存器碎片与 '寄存器代数'")
