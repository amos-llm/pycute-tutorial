"""pycute-tutorial 共享工具。

直接使用系统已安装的 pycute (不修改 sys.path), 提供:
  1. 打印/验证的小工具 (banner, show_grid, assert_same, exercise ...);
  2. 两个教程专用 helper:
       - tile_to_shape : 模仿 cute::tile_to_shape (pycute 没有这个便捷函数)
       - SwizzledLayout: 模仿 cute 的 "Swizzle o Layout" 组合布局
"""

from pycute import Layout, LayoutBase, is_tuple

# ---------------------------------------------------------------- 打印工具


def banner(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def step(msg):
    print(f"\n--- {msg}")


def show_grid(layout, title=None):
    """把一个 2D (非嵌套) layout 画成偏移量矩阵, 直观看出数据是怎么排的。"""
    assert is_tuple(layout.shape) and len(layout.shape) == 2
    assert not is_tuple(layout.shape[0]) and not is_tuple(layout.shape[1]), (
        "show_grid 只支持两个 mode 都是整数的 layout"
    )
    R, C = layout.shape
    if title:
        print(title)
    w = max(len(str(layout(R - 1, c))) for c in range(C))
    for r in range(R):
        print("   " + " ".join(f"{layout(r, c):>{w}}" for c in range(C)))


# ---------------------------------------------------------------- 验证工具


def domain_2d(R, C):
    return ((r, c) for r in range(R) for c in range(C))


def assert_same(f, g, domain, label):
    """断言两个映射 f(crd)==g(crd) 在整个 domain 上相等。"""
    for crd in domain:
        assert f(*crd) == g(*crd), f"{label}: f{crd}={f(*crd)} != g{crd}={g(*crd)}"


def check(name, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return ok


def exercise(name, fn):
    """运行一个练习; 未填写时 fn 会 raise NotImplementedError 或 "请填写" 断言。"""
    try:
        fn()
    except NotImplementedError:
        print(f"  [--] {name}: 未完成 (按提示填写代码后重跑)")
    except AssertionError as e:
        if "请填写" in str(e):
            print(f"  [--] {name}: 未完成 (按提示填写代码后重跑)")
        else:
            print(f"  [XX] {name}: 答案不对 -> {e}")
    except Exception as e:
        print(f"  [XX] {name}: 输入有问题 ({type(e).__name__}: {e})")


# ---------------------------------------------------------------- 教程专用 helper


def tile_to_shape(atom, shape):
    """模仿 cute::tile_to_shape (2D 版)。

    把 atom (s0,s1):(d0,d1) 沿两个 mode 平铺, 直到填满 shape。
    要求: shape[i] % s_i == 0, 且 tile 按 "先 mode0 方向, 再 mode1 方向" 排列
    (即 C++ tile_to_shape 的行为: tile 索引按 mode0 优先)。

    结果: ((s0,c0),(s1,c1)):((d0, K), (d1, c0*K)), 其中 c_i = shape_i/s_i,
    K 是 atom 的 cosize (atom 占用的连续空间大小)。
    """
    s0, s1 = atom.shape
    d0, d1 = atom.stride
    assert not is_tuple(s0) and not is_tuple(s1)
    assert shape[0] % s0 == 0 and shape[1] % s1 == 0
    c0, c1 = shape[0] // s0, shape[1] // s1
    K = (s0 - 1) * d0 + (s1 - 1) * d1 + 1  # atom cosize
    return Layout(((s0, c0), (s1, c1)), ((d0, K), (d1, c0 * K)))


class SwizzledLayout(LayoutBase):
    """模仿 cute 的组合布局  Swizzle o Layout。

    pycute 的 composition() 不支持 Swizzle 作为左操作数, 而 FA2 的
    SmemLayoutAtomQ 正是 composition(Swizzle<3,3,3>, (8,64):(64,1))。
    这里按 ComposedLayout 的语义实现: 先查 base layout, 再对偏移做 swizzle。
    """

    def __init__(self, swz, layout):
        self.swz = swz
        self.layout = layout
        self.shape = layout.shape
        self.stride = layout.stride

    def __call__(self, *args):
        return self.swz(self.layout(*args))

    def __str__(self):
        return f"{self.swz} o {self.layout}"

    def __repr__(self):
        return f"SwizzledLayout({self.swz!r},{self.layout!r})"
