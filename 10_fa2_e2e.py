"""第 10 章: 端到端 —— 追一个元素穿过 FA2 的前向 kernel

配置 (与 FA2 hdim=128 非 causal 的 sm8x 标准编译配置一致,
见 flash_fwd_launch_template.h: run_mha_fwd_hdim128):
    kBlockM=128, kHeadDim=128, kNWarps=4 (128 线程), kBlockKSmem=64, kSwizzle=3
    Q tile: 128 行 x 128 k (fp16), seqlen_q = 200, m_block = 0
    smem  : atom (8,64):(64,1) + Swizzle<3,3,3>, 平铺到 (128,128)
    gmem 拷贝: GmemLayoutAtom = (16,8):(8,1) + value (8):(1), 8 行块 x 2 k 页
    mma   : 4 warps 沿 M, 每 warp 两个 16 行带 (MMA_M=2); K 方向 8 个 k16 块

两个阶段的 layout 代数:
    gmem(m,k) --[copy 线程切分]--> smem 偏移 (swizzled) --[mma 碎片]--> 寄存器
    全程 "数据不动, 换坐标", 本章把每个环节写出来并验证。

运行: python3 10_fa2_e2e.py
"""

from common import banner, check, exercise, step, tile_to_shape
from pycute import idx2crd, Layout, Swizzle

banner("第 10 章: 端到端 —— 追一个元素穿过 FA2 的前向 kernel")

step("10.1 定义三个空间的 layout")
swz = Swizzle(3, 3, 3)

# (1) gmem: Q 按 (seqlen, d) 行主序
mQ = Layout((200, 128), (128, 1))

# (2) smem: swizzled tile
smem_base = tile_to_shape(Layout((8, 64), (64, 1)), (128, 128))


def smem_off(m, k):
    return swz(smem_base(m, k))


# (3) gmem 拷贝: 线程切分 (128 线程)
copy_atom = Layout((16, 8), (8, 1))  # 128 线程 -> (行块内行, 列块)


def loader(m, k):
    """gmem 元素 (m,k) 由谁搬: 返回 (thread, 行块, k页, 值内偏移)。"""
    t = 8 * (m % 16) + (k % 64) // 8
    return t, m // 16, k // 64, k % 8


# (4) mma: A 碎片 (m16n8k16), 4 warps, 每 warp 两个 16 行带 (mm=0,1, 相距 64 行)
# 值三元组按 (cp, rh, kh) 记, 与第 9 章 PTX 寄存器序一致: v = cp + 2*rh + 4*kh
def mma_owner(row, k):
    """smem 元素 (row,k) 被哪个 mma 线程的哪个值读取。返回 (thread, mm, kb, 值)。"""
    mm, r64 = row // 64, row % 64
    w, r16 = r64 // 16, r64 % 16
    g, rh = r16 % 8, r16 // 8
    kb, koff = k // 16, k % 16
    tig, cp, kh = (koff % 8) // 2, koff % 2, koff // 8
    return 32 * w + 4 * g + tig, mm, kb, (cp, rh, kh)


step("10.2 阶段 1: gmem -> smem 是双射")
offs = {smem_off(m, k) for m in range(128) for k in range(128)}
print(
    f"128x128 个 (m,k) 映射到 {len(offs)} 个不同 smem 偏移: "
    f"{'OK (双射)' if len(offs) == 128 * 128 else 'FAIL'}"
)

step("10.3 阶段 2: smem -> 寄存器 恰好覆盖一次")
seen = set()
for t in range(128):
    w, lane = t // 32, t % 32
    g, tig = lane // 4, lane % 4
    for mm in range(2):
        for kb in range(8):
            for v in range(8):
                cp, rh, kh = idx2crd(v, (2, 2, 2))
                seen.add(
                    (64 * mm + 16 * w + g + 8 * rh, kb * 16 + tig * 2 + cp + 8 * kh)
                )
print(
    f"128 线程 x 2 行带 x 8 块 x 8 值 = {128 * 2 * 8 * 8} 个 (row,k), "
    f"覆盖 {len(seen)} 个不同位置: "
    f"{'OK (恰好一次)' if len(seen) == 128 * 128 else 'FAIL'}"
)

step("10.4 追踪元素 Q(5, 20)")
m, k = 5, 20
t, mt, kp, v = loader(m, k)
t2, mm, kb, vv = mma_owner(m, k)
# mma 线程 t2 读该值时的 smem 位置:
#   row = 64mm + 16w + g + 8rh, k = 16kb + 2tig + cp + 8kh
w2, lane2 = t2 // 32, t2 % 32
read_pos = (
    64 * mm + 16 * w2 + lane2 // 4 + 8 * vv[1],
    kb * 16 + (lane2 % 4) * 2 + vv[0] + 8 * vv[2],
)
print(f"""
  gmem 偏移           = mQ(5,20)        = {mQ(5, 20)}
  谁从 gmem 搬        = 线程 {t}, 行块 {mt}, k页 {kp}, 值 #{v}
  搬进 smem 的偏移    = swz(smem(5,20)) = {smem_off(m, k)}
  谁用 ldmatrix 读    = mma 线程 {t2}, 行带 {mm}, k块 {kb}, 值 {vv}
  它读的 smem 位置    = {read_pos}
  两者一致            = {read_pos == (m, k)}
""")

step("10.5 全量一致性: 两条路径都能无损还原 (m,k)")
ok = True
for m in range(128):
    for k in range(128):
        # load 路径: (m,k) -> (t, mt, kp, v) -> 还原
        t, mt, kp, v = loader(m, k)
        m1, k1 = mt * 16 + t // 8, kp * 64 + (t % 8) * 8 + v
        # mma 路径: (m,k) -> (t2, mm, kb, (cp,rh,kh)) -> 还原
        t2, mm, kb, (cp, rh, kh) = mma_owner(m, k)
        w2, lane2 = t2 // 32, t2 % 32
        g2, tig2 = lane2 // 4, lane2 % 4
        m2, k2 = 64 * mm + 16 * w2 + g2 + 8 * rh, kb * 16 + tig2 * 2 + cp + 8 * kh
        if (m1, k1) != (m, k) or (m2, k2) != (m, k):
            ok = False
print("load 与 mma 两条路径对每个 (m,k) 都能无损还原:", "OK" if ok else "FAIL")

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 手算 Q(37, 100) 的 loader 四元组 (t, mt, kp, v)。
    your_loader = None  # <-- 填 (t, mt, kp, v)
    assert your_loader is not None, "请填写 your_loader"
    check(f"练习1: your_loader={your_loader}", your_loader == loader(37, 100))


def ex2():
    # 练习 2: 验证阶段 1 的双射性 (参照 10.2, 自己写一遍集合推导)。
    offs = set()
    for m in range(128):
        for k in range(128):
            offs.add(smem_off(m, k))
    check("练习2: 阶段1 双射", len(offs) == 128 * 128)


def ex3():
    # 练习 3: 手算 Q(100, 77) 的 mma_owner (线程, 行带, k块, 值), 验证。
    # (值三元组是 (cp, rh, kh), 与第 9 章 PTX 寄存器序 v = cp + 2*rh + 4*kh 一致)
    your_mma = None  # <-- 填 (thread, mm, kb, (cp,rh,kh))
    assert your_mma is not None, "请填写 your_mma"
    got = mma_owner(100, 77)
    check(f"练习3: your_mma={your_mma}", your_mma == got)


def ex4():
    # 练习 4 (扩展): d=96 是 FA2 真实支持的 head dim (kHeadDim=96,
    # kBlockKSmem=32, kSwizzle=2, 见 kernel_traits.h)。此时 smem atom 变成
    # (8,32):(32,1) + Swizzle(2,3,3), 平铺到 (128,96)。验证仍是双射。
    swz2 = Swizzle(2, 3, 3)
    smem96 = tile_to_shape(Layout((8, 32), (32, 1)), (128, 96))
    offs96 = {swz2(smem96(m, k)) for m in range(128) for k in range(96)}
    check("练习4: (128,96) swizzled 双射", len(offs96) == 128 * 96)


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)
exercise("练习4", ex4)

banner("全部 10 章完成!")
print(
    """
下一步建议:
  1. 回到 csrc/flash_attn/src/flash_fwd_kernel.h, 把每一行 partition/copy 注释到
     "哪个 layout 代数操作" 为止 (本教程每章开头都给了对应代码行)。
  2. 用 csrc/cutlass/examples/03_visualize_layout/register_layout.cu 可视化 TV 布局。
  3. 读 02_layout_algebra.md 的剩余部分 """
    """(by-mode composition / composition tilers / complement)。
"""
)
