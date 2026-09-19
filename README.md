# CuTe Layout 代数教程 (pycute 版, 面向 FA2)

一套用 **pycute** (CUTLASS 仓库里的 CuTe Python 实现, 无需编译 CUDA) 循序渐进学习
**layout 代数** 的教程, 每一章都对应 FlashAttention-2 代码里真实出现的操作。

学完你应能读懂 `csrc/flash_attn/src/kernel_traits.h` 里的每一行 `Layout<...>` 声明,
以及 `flash_fwd_kernel.h` 里所有 `partition_*` / `composition` / `local_tile` 调用。

## 运行方式

```bash
python3 01_layout_is_function.py
python3 10_fa2_e2e.py
```

- 依赖系统已安装的 `pycute` (随 CUTLASS / `nvidia-cutlass` 提供,
  即 `python3 -c "import pycute"` 能通过), 不需要改 `sys.path`。
- 文件里的**练习**默认未完成 (打印 `[--] ... 未完成`), 按提示填写 `your_xxx = ...` 后重跑:
  通过打印 `[OK]`; 数值答案不对打印 `[FAIL] 练习X: ...`; 用 assert 判分的练习 (05-ex3、07 等)
  答案不对打印 `[XX] 答案不对 -> ...`。另有 6 个练习是预写好的自动验证 (无空白, 恒 [OK]), 展示判分方式。
- 参考答案在 `solutions/` 目录 (对应章节), 建议先自己想 5-10 分钟再偷看。

## 章节地图

| 章 | 主题 | 关键代数操作 | FA2 对应代码 |
|---|------|-------------|-------------|
| 01 | Layout 就是函数: 坐标 → 偏移 | `Layout(shape, stride)`, `idx2crd` | 所有 `Layout<Shape,Stride>` |
| 02 | 二维与嵌套 mode | 嵌套 shape/stride, 切片 `L(r, None)` | `GmemLayoutAtom` 等 |
| 03 | 化简 | `coalesce` (4 条合并规则) | 各种 `coalesce` 后的布局 |
| 04 | **复合 (核心)** | `composition(A, B) = A o B` | 一切 "换视角" |
| 05 | 乘与除 (平铺/切块) | `logical_product`, `logical_divide`, `zipped_divide`, `tile_to_shape` | `SmemLayoutQ`, `local_tile` |
| 06 | Swizzle (smem 无冲突) | `Swizzle<B,M,S>` = 地址位异或 | `SmemLayoutAtomQ`, `kSwizzle` |
| 07 | 免费的转置 | `composition(SmemLayoutKV, (d,n) 行主序)` | `SmemLayoutVtransposed` |
| 08 | identity 与谓词 | identity tensor + 同样的切分 | `make_identity_tensor`, `tQpQ` |
| 09 | MMA 寄存器碎片 | `logical_divide` + mode 重排 | `convert_layout_acc_Aregs` |
| 10 | 端到端: 追一个元素 | 以上全部串起来 | `flash_fwd_kernel.h` 全流程 |

## 学习建议

1. **不要跳过手算**。每章的练习都要求先手算再验证 —— layout 代数的直觉
   来自 "我能自己推出来", 而不是 "程序说它对"。
2. **每章结尾都指向 FA2 代码**。学完一章, 去读对应的源码行, 把抽象符号对上具体常量。
3. 卡住了就去 `flash-attention/csrc/cutlass/media/docs/cpp/cute/02_layout_algebra.md`
   (相对 flash-attention 仓库根) 看官方推导 (本教程第 3/4/5 章的公式都来自它)。
4. 第 9 章是 FA2 最精妙的部分, 值得花两遍: 先理解 C 碎片, 再理解
   "C 碎片的值顺序 == A 碎片的前 4 个值顺序" 这个巧合。
5. 第 10 章跑完后, 试着把 `loader()` / `mma_owner()` 换成 FA2 的其他真实配置
   (如 sm8x causal 的 kBlockM=64, 或 hdim96 的 kSwizzle=2), 检查覆盖性是否还成立。

## 文件结构

```
pycute-tutorial/
├── README.md            # 本文件
├── common.py            # 共享工具: banner/show_grid/tile_to_shape/SwizzledLayout
├── 01_layout_is_function.py
├── 02_2d_and_modes.py
├── 03_coalesce.py
├── 04_composition.py
├── 05_product_and_divide.py
├── 06_swizzle.py
├── 07_free_transpose.py
├── 08_identity_and_predicates.py
├── 09_mma_fragments.py
├── 10_fa2_e2e.py
└── solutions/           # 各章练习的参考答案
    ├── solutions_01.py
    ├── ...
    └── solutions_10.py
```
