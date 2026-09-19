"""第 6 章: Swizzle —— 用位异或消灭 smem bank conflict

目标: 理解 FA2 smem atom 里的 Swizzle<3,3,3>, 并验证它确实消除了
ldmatrix (8 行 x 16B) 读取时的 bank conflict。

背景知识:
  - smem 有 32 个 bank, 每个 bank 每周期提供 4 字节。
  - 一个 fp16 元素占 2 字节, 所以 元素 i 所在的 bank = (i // 2) % 32。
  - ldmatrix 一次读 8 行, 每行 16B (8 个 fp16) = 4 个 bank。
    8 行 x 4 bank = 32 bank, 若恰好各用一次 => 无冲突。

FA2 代码 (kernel_traits.h):
    kSwizzle = kBlockKSmem == 32 ? 2 : 3
    SmemLayoutAtomQ =
        composition(Swizzle<kSwizzle,3,3>, (8, kBlockKSmem):(kBlockKSmem,1))

运行: python3 06_swizzle.py
"""

from common import banner, check, exercise, step, SwizzledLayout
from pycute import Layout, Swizzle
from collections import Counter

banner("第 6 章: Swizzle —— 用位异或消灭 smem bank conflict")

step("6.1 Swizzle 是地址上的位异或函数")
print("""
  Swizzle<BBits, MBase, SShift> (cute 记号; FA2 用 Swizzle<3,3,3>):
    以 Swizzle<3,3,3> 为例 (BBits=3, MBase=3, SShift=3):
      bit 位:   9 及以上  | 8   7   6  | 5   4   3  | 2   1   0
                   ... x   |  Y  Y  Y  |  Z  Z  Z  |  x   x   x
      YYY = bits[6:9)  (参与异或的源),  ZZZ = bits[3:6)  (被改写的位置)
  pycute: Swizzle(bits, base, shift):
    yyy 掩码 = bits 个 1 左移 (base + shift)
    swz(x) = x ^ ((x & yyy_msk) >> shift)
""")
swz = Swizzle(3, 3, 3)
print(f"swz(0)   = {swz(0)}")
print(
    f"swz(96)  = {swz(96)}   # 96 = 0b01100000, bits[6:9)=001 -> ^ 0b001000(8) -> 104"
)
print(
    f"swz(352) = {swz(352)}  "
    f"# 352 = 0b101100000, bits[6:9]=101 -> ^ 0b0101000(40) -> 328"
)

step("6.2 FA2 的 smem atom: (8, 64):(64,1) + Swizzle<3,3,3>")
# 64 个 fp16 = 128 字节 = 一行; 32 个 bank 正好一行
base = Layout((8, 64), (64, 1))
atom_swz = SwizzledLayout(swz, base)  # 对应 FA2 的 composition(Swizzle, base)
print("base atom      =", base)
print("swizzled atom  =", atom_swz)
print("atom_swz(1, 32) =", atom_swz(1, 32), " = swz(1*64+32) =", swz(96))


def banks_of_16b_chunk(row, col8, width):
    """一行里 col8 开始的 8 个 fp16 (16B) 覆盖的 4 个 bank。"""
    w = atom_swz(row, col8) // 2
    return [(w + i) % 32 for i in range(4)]


step("6.3 ldmatrix 模式: 8 个连续行, 读同一列块")
for col8 in (0, 32):
    banks = []
    for r in range(8):
        banks += banks_of_16b_chunk(r, col8, 64)
    cnt = Counter(banks)
    start_banks = [min(banks_of_16b_chunk(r, col8, 64)) for r in range(8)]
    print(f"col8={col8}: 各行的起始 bank = {start_banks}")
    print(
        f"         32 个 bank 使用次数全为 1: {all(v == 1 for v in cnt.values())}  "
        f"(无冲突)"
    )

step("6.4 不用 swizzle 会怎样?")
for col8 in (0, 32):
    start_banks = [(r * 64 + col8) // 2 % 32 for r in range(8)]
    print(f"col8={col8}: 起始 bank = {start_banks}  (8 行撞同一个 bank -> 8 路冲突)")

step("6.5 swizzle 是双射 (bijection)")
img = [swz(x) for x in range(512)]
print("swz 把 [0,512) 映射成它的一个排列:", sorted(img) == list(range(512)))
print("=> 数据总量/形状都不变, 只是地址被重排; 这就是 '免费' 的代价模型。")

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: 手算 swz(64) 和 swz(160) (Swizzle(3,3,3)), 填数字验证。
    your_64 = None  # <-- 填
    your_160 = None  # <-- 填
    assert your_64 is not None and your_160 is not None, "请填写两个空"
    ok = your_64 == swz(64) and your_160 == swz(160)
    check(f"练习1: your_64={your_64}, your_160={your_160}", ok)


def ex2():
    # 练习 2: kBlockKSmem=32 时 FA2 用 Swizzle<2,3,3>, atom = (8,32):(32,1)。
    # (a) 手算 row=3, col8=8 的元素起点 swz2(3*32+8) 落在哪个 bank (填 your_bank);
    # (b) 再验证: 对每个 16B 列块 col8 in {0,8,16,24}, 8 行 ldmatrix 读取无冲突。
    swz2 = Swizzle(2, 3, 3)

    def banks2(row, col8):
        w = (swz2(row * 32 + col8)) // 2
        return [(w + i) % 32 for i in range(4)]

    your_bank = None  # <-- 填 swz2(104)//2 % 32
    assert your_bank is not None, "请填写 your_bank"
    ok = your_bank == (swz2(3 * 32 + 8) // 2) % 32
    for col8 in (0, 8, 16, 24):
        banks = []
        for r in range(8):
            banks += banks2(r, col8)
        if not all(v == 1 for v in Counter(banks).values()):
            ok = False
    check("练习2: 单点 bank 手算 + 32 宽行全列块无冲突", ok)


def ex3():
    # 练习 3: 手算 swz2(112) (row=3, col8=16 的元素起点), 验证。
    swz2 = Swizzle(2, 3, 3)
    your_answer = None  # <-- 填
    assert your_answer is not None, "请填写 your_answer"
    check("练习3: swz2(112)", your_answer == swz2(112))


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3", ex3)

banner("第 6 章结束。下一章: 免费的转置 (V^T)")
