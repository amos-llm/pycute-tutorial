"""第 4 章: composition —— layout 代数的核心

目标: 理解并会用函数复合。这是 FA2 里用得最多的操作:
"换一种看法" = 复合一个新的 layout。

定义:
    R = composition(A, B)   即  R = A o B
    R(c) = A(B(c))
    B 定义 R 的定义域 (domain), A 是 "被重排" 的那个。

运行: python3 04_composition.py
"""

from common import banner, check, exercise, step
from pycute import composition, idx2crd, Layout

banner("第 4 章: composition —— layout 代数的核心")

step("4.1 官方文档的例子 (02_layout_algebra.md 第 1 个例子)")
A = Layout((6, 2), (8, 2))
B = Layout((4, 3), (3, 1))
R = composition(A, B)
print("A =", A)
print("B =", B)
print("R = A o B =", R)
print("逐个验证 R(c) == A(B(c)):")
print("  c  B(c)        A(B(c))   R(c)")
for c in range(12):
    bc = B(c)
    ab = A(*idx2crd(bc, (6, 2))) if isinstance(bc, int) else A(bc)
    print(f"  {c:>2}  {bc:<10} {ab:<9} {R(c)}")

step("4.2 特殊形式: A o (s:d) = '取 A 的每隔 d 个元素, 取 s 个'")
A2 = Layout((8, 4), (4, 1))
R2 = composition(A2, Layout(4, 2))
print("A2 =", A2)
print("A2 o (4):(2) =", R2, "  # 取 A2 的 1D 序列 0,2,4,6 号元素")

step("4.3 复合的兼容性 (什么时候 A o B 有定义)")
print("""
  CuTe 官方 (02_layout_algebra.md "Computing Composition") 有两条除法条件:
    ① Stride Divisibility: 对 A (先 coalesce) 的每个 mode (s_i, d_i),
       B 的当前步长 d 须满足  d % s_i == 0  或  d < s_i
    ② Shape Divisibility: 每个 mode 上 "取的个数" 须整除 B 剩余的 shape
  一句话: B 的步长必须能 "踩" 在 A 的 mode 边界上, 否则复合无定义。
  注意: pycute 的 assert 比 C++ 更严 (要求 s_i 与 d 互整除), 所以个别
  在 C++ 合法的复合, pycute 会 assert。
  另一个坑: 条件没完全满足时, C++ (static shape) 会触发 Shape 条件,
  但 pycute 可能静默截断, 如 composition((4,4):(16,1), (6):(1)) -> 4:16
  (size=4, 不是 B 的 size 6)。所以养成习惯: size(R) != size(B) 就要怀疑。
  下面的例子会触发 pycute 的 assert (所以这里用 try 包住):
""")
try:
    composition(Layout((3, 4), (4, 1)), Layout(2, 2))  # 步长 2 踩不进 shape 3
    print("没有报错 (不应该发生)")
except AssertionError:
    print("  composition((3,4):(4,1), (2):(2)) -> assert 失败, 正如预期")

step("4.4 FA2 铺垫: '转置' 就是复合 (第 7 章细讲)")
# smem 里 V 按 (n=64, d=128) 存, d 连续; 想按 (d, n) 读:
baseKV = Layout((64, 128), (128, 1))  # sV(n,d)
B = Layout((128, 64), (64, 1))  # (d,n) 行主序 "读法"
Vt = composition(baseKV, B)
print("sV  =", baseKV, " (n,d)")
print("读法B =", B, " (d,n)")
print("Vt = sV o B =", Vt, " (d,n)")
print("验证: Vt(i,j) == sV(j,i) 对所有 (i,j) 成立:")
assert all(Vt(i, j) == baseKV(j, i) for i in range(128) for j in range(64))
print("  True")

# ================================================================ 练习

step("练习")


def ex1():
    # 练习 1: A = (3,4):(4,1) (行主序 12), B = (2,3):(6,1)。
    # (a) 用 composition 求 R = A o B, 打印它;
    # (b) 写个循环暴力验证 R(p,q) == A(B(p,q)) 对所有 (p,q)。
    A = Layout((3, 4), (4, 1))
    B = Layout((2, 3), (6, 1))
    R = None  # <-- 填 composition(A, B)
    assert R is not None, "请填写 R"
    ok = all(R(p, q) == A(B(p, q)) for p in range(2) for q in range(3)) and R == Layout(
        (2, 3), (2, 4)
    )
    check(f"练习1: A o B = {R}", ok)


def ex2():
    # 练习 2: 求 (8,4):(4,1) o (4):(2) 的结果 (先手算: 取 1D 序列的第 0,2,4,6 个元素)。
    your_answer = None  # <-- 填 Layout
    assert your_answer is not None, "请填写 your_answer"
    check(
        "练习2: (8,4):(4,1) o (4):(2)",
        your_answer == composition(Layout((8, 4), (4, 1)), Layout(4, 2)),
    )


def ex3():
    # 练习 3 (思考题): 为什么 pycute 里 A = (3,4):(4,1), B = (2):(2) 的复合会 assert?
    # 提示: 看 4.3 条件 ①, A 的非末 mode shape 3 与 B 的 stride 2 的关系,
    # 以及 pycute layout.py 的 assert 表达式 (要求互整除)。先手算 A(0), A(2) 落在哪,
    # 再想 C++ 会怎么做 (完整解析在 solutions/solutions_04.py)。
    print("你的解释: <在这里写一句话>")


exercise("练习1", ex1)
exercise("练习2", ex2)
exercise("练习3(思考)", ex3)

banner("第 4 章结束。下一章: 乘与除 (product / divide)")
