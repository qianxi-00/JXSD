"""
对应课案章节：PyTorch / 数据（Dataset 与 DataLoader）

本节知识点：
    1.  为什么需要 Dataset / DataLoader：把「单个样本怎么读」和「怎么组装成 batch」两件事分开
    2.  自定义 Dataset：必须实现 `__init__` / `__len__` / `__getitem__` 三个方法，缺一不可
    3.  用 DataLoader(batch_size=16, shuffle=True) 遍历，打印每个 batch 的实际形状
    4.  `drop_last=True/False` 对「最后不满一个 batch」的影响（100 条样本 / batch_size=16 = 6 满 + 1 残）
    5.  `shuffle=True/False` 对取数顺序的影响（打印前 8 个样本的标签对比）
    6.  `num_workers` 在 Windows 上的注意事项：spawn 启动方式 + `if __name__ == "__main__"` 保护
    7.  `pin_memory` 的用途（锁页内存加速 CPU→GPU 拷贝），以及 CPU 版下它没有收益的原因
    8.  自定义 `collate_fn`：把变长序列 padding 到同一长度，并同时产出 attention mask
    9.  `TensorDataset`：把多个张量按第 0 维对齐，快速包成 Dataset
    10. `len(dataset)` 与 `len(dataloader)` 的关系（用 math.ceil 推出通用公式）
    11. 用 `Subset` / `random_split` 做训练集 / 验证集切分（并说明为什么必须切分）
    12. 手写 `MyDataLoader`：只用「索引列表 + 列表推导」实现 shuffle + 分批，和官方 DataLoader 对比
    13. 多进程加载 + 真实训练循环：`num_workers=0` vs `1` 的耗时对比，以及课案那五行的训练循环
    14. 画图：drop_last 的 batch 数对比、padding mask 热力图、两种加载器的数据覆盖验证

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\01_PyTorch基础\\03_数据集与DataLoader.py'
"""

# ---------------------------------------------------------------------------
# 统一环境初始化：目录定位 + 无界面绘图后端
# ---------------------------------------------------------------------------
from pathlib import Path

_HERE = Path(__file__).resolve().parent     # 当前脚本所在目录
_ROOT = _HERE.parent                        # Deep_Learning 根目录

import matplotlib
matplotlib.use("Agg")                       # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = _ROOT / "output"               # 所有图片统一输出到这里
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 正式内容开始
# ---------------------------------------------------------------------------
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

class CustomDataset(Dataset):
    """自定义 Dataset 示例：用 torch.randn 造特征，用线性可分规则造二分类标签。

    说明：本机禁止联网/下载数据，所以数据一律用随机数生成。
          真实项目里这里会换成「读 csv / 读图片 / 读 parquet」，但对 DataLoader 完全透明。
    """

    def __init__(self, n_samples: int = 100, n_features: int = 10, seed: int = 42):
        # __init__ 只做「准备工作」：把数据准备好、算好元信息。
        # 工程建议：如果要读大文件，不要在 __init__ 里把全部内容读进内存，
        #           而是只存「文件路径列表」，真正的读取放到 __getitem__ 里按需做（lazy loading）。
        generator = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n_samples, n_features, generator=generator)   # 特征 (n, d)
        # 标签规则：前两个特征之和 > 0 就是类别 1，否则类别 0。
        # 这是一个「线性可分」的问题，模型很容易学到，方便验证数据管线是否正确。
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).long()                  # 标签 (n,)
        self.n_features = n_features

    def __len__(self) -> int:
        # __len__ 必须返回「样本总数」。DataLoader 靠它算 batch 个数、做 shuffle 采样。
        # 注意：返回的必须是 Python int，不能是张量。
        return len(self.X)

    def __getitem__(self, idx: int):
        # __getitem__ 必须返回「第 idx 条样本」。idx 可以是 int，也可以是 list/张量（批量索引）。
        # 返回的通常是 (特征, 标签) 元组；DataLoader 会把这个元组自动转成「按位置分组的 batch」。
        # 也就是说：返回 (x, y) → DataLoader 给出 (batch_X, batch_y)，这是约定俗成的写法。
        return self.X[idx], self.y[idx]


def _compare_worker_reads(dataset, batch_size, num_workers):
    """用同一份数据集分别以 num_workers 和 0 读一遍，返回 (标签, 是否完全一致)。

    这个函数是模块级的（不是 main 里的闭包），所以 dataset 能被正常 pickle 给子进程。
    """
    labels_multi = torch.cat([y for _, y in DataLoader(dataset, batch_size=batch_size,
                                                       shuffle=False, num_workers=num_workers)])
    labels_single = torch.cat([y for _, y in DataLoader(dataset, batch_size=batch_size,
                                                        shuffle=False, num_workers=0)])
    return labels_multi, bool(torch.equal(labels_multi, labels_single))


class VariableLengthSeqDataset(Dataset):
    """变长序列数据集：每条样本长度不同，模拟一批长度不一的文本 token。"""

    def __init__(self, lengths, vocab_size=50, seed=7):
        generator = torch.Generator().manual_seed(seed)
        # 每条序列是一个随机 token id 序列（1..vocab_size-1，故意避开 0，因为 0 要用作 padding）
        self.sequences = [
            torch.randint(low=1, high=vocab_size, size=(L,), generator=generator)
            for L in lengths
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]          # 只返回一条变长序列（长度各不相同）


def pad_collate_fn(batch):
    """自定义 collate_fn：把一批变长序列 pad 到本 batch 的最长长度，并返回 mask。

    参数 batch：是一个 list，元素是 Dataset.__getitem__ 的返回值。
                这里每个元素是一个形状 (L_i,) 的 1 维张量。
    返回：(padded, mask)
        padded —— (batch, max_len) 的 int64 张量，短序列右侧补 0
        mask   —— (batch, max_len) 的 bool 张量，True=真实 token，False=padding
    """
    lengths = [seq.shape[0] for seq in batch]
    max_len = max(lengths)                            # 本 batch 的动态最大长度
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, seq in enumerate(batch):
        L = seq.shape[0]
        padded[i, :L] = seq                           # 左侧对齐填入真实 token
        mask[i, :L] = True                            # 真实位置标 True，其余保持 False
    return padded, mask


class MyDataLoader:
    """极简 DataLoader：只用「索引列表 + 列表推导」实现 shuffle + batch。

    刻意不依赖 torch.utils.data.DataLoader，用于理解官方实现的核心思路。
    为简单起见只支持 (特征, 标签) 这种二元组样本，且单进程（无多进程加载）。
    """

    def __init__(self, dataset, batch_size=16, shuffle=False, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        # 预先算好 batch 个数，逻辑与官方公式一致
        n = len(dataset)
        if drop_last:
            self.n_batches = n // batch_size
        else:
            self.n_batches = (n - 1) // batch_size + 1 if n > 0 else 0

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))                      # 步骤 1：所有样本下标
        if self.shuffle:                              # 步骤 2：打乱下标
            # 用 random.Random(seed) 生成独立随机源，避免污染全局随机状态
            random.Random(self.seed).shuffle(indices)
        # 步骤 3：按 batch_size 切片 → 列表推导取样本 → 按位置转置成 batch
        for start in range(0, n, self.batch_size):
            chunk = indices[start:start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                continue                              # 丢弃最后一个不满的 batch
            samples = [self.dataset[i] for i in chunk]      # list[(x_i, y_i)]
            # zip(*samples) 把 [(x1,y1), (x2,y2)] 转置成 [(x1,x2), (y1,y2)]
            xs, ys = zip(*samples)
            yield torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0)


def _make_model():
    """建一个两层小网络。每次对比前都重建，保证 num_workers 不同时起点一致。"""
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )


def main():
    """把全部演示逻辑收进函数。

    为什么要这样组织？Windows 上 DataLoader 的 num_workers>0 用 spawn 启动子进程，
    子进程会**重新导入本模块**（此时模块名是 __mp_main__，不是 __main__）。
    所有需要被 pickle 的类（CustomDataset 等）必须定义在模块级，子进程才 import 得到；
    而所有演示/训练代码放进 main() 并加 __main__ 保护，子进程重新导入时就不会重复执行。
    """
    print("=" * 78)
    print("03 数据集与 DataLoader：自定义 Dataset / batch 迭代 / collate_fn / 手写加载器")
    print("=" * 78)
    print(f"PyTorch 版本：{torch.__version__}    CUDA 可用：{torch.cuda.is_available()}")
    print()


    # ===========================================================================
    # 【1】为什么要 Dataset + DataLoader：职责分离
    # ===========================================================================
    print("-" * 78)
    print("【1】Dataset 与 DataLoader 的分工")
    print("-" * 78)

    # ---- 原理 ----
    # 训练时要反复做三件事：读一条样本 → 凑一批 → 送进模型。
    # 如果全写在一个循环里，代码会和「数据格式」死死绑在一起，换个数据集就要重写训练代码。
    # 官方抽象把它拆成两个各司其职的角色：
    #
    #   Dataset（数据集）
    #       —— 只回答一个问题：「第 i 条样本长什么样？」
    #          约定实现 __len__（一共多少条）和 __getitem__（第 i 条是什么）。
    #          它不关心 batch、不关心打乱、不关心并行，可以说是「随机的下标访问」。
    #
    #   DataLoader（数据加载器）
    #       —— 只回答一个问题：「给我一批一批的数据。」
    #          它负责：按 batch_size 分组、是否 shuffle、用几个子进程读(num_workers)、
    #                  怎么把多条样本拼成一个 batch(collate_fn)、是否锁页(pin_memory)。
    #
    # 好处：换数据集只改 Dataset；调 batch 策略只改 DataLoader；训练循环一个字都不用动。
    # 这就是课案里那张参数表的由来。




    dataset = CustomDataset(n_samples=100, n_features=10)

    print("自定义 CustomDataset(n_samples=100, n_features=10)：")
    print(f"  len(dataset)      = {len(dataset)}（样本总数）")
    print(f"  dataset[0]        = (特征 {tuple(dataset[0][0].shape)}, 标签 {dataset[0][1].shape}={dataset[0][1].item()})")
    print(f"  dataset[0][0] 前 5 个特征 = {[round(v, 4) for v in dataset[0][0][:5].tolist()]}")
    print(f"  标签分布：类别 0 有 {int((dataset.y == 0).sum())} 条，类别 1 有 {int((dataset.y == 1).sum())} 条")

    # Dataset 支持「批量下标」：DataLoader / 我们自己写的加载器都靠这个特性
    batch_idx = [3, 7, 11]
    sub_X, sub_y = dataset[batch_idx]           # idx 是 list → 返回的是堆叠后的张量
    print(f"  dataset[[3, 7, 11]] → X shape={tuple(sub_X.shape)}, y shape={tuple(sub_y.shape)}（下标可以是列表）")
    print()


    # ===========================================================================
    # 【2】DataLoader 基本用法：遍历并打印每个 batch 的形状
    # ===========================================================================
    print("-" * 78)
    print("【2】DataLoader(batch_size=16, shuffle=True) 遍历：每个 batch 的真实形状")
    print("-" * 78)

    # ---- 原理 ----
    # DataLoader 是可迭代对象（实现了 __iter__），每次 for 循环会：
    #     1) 生成一个样本下标序列（shuffle=True 时用 RandomSampler 打乱）
    #     2) 按 batch_size 切块，最后一块可能不满
    #     3) 对每块调用 collate_fn（默认是 default_collate：把 list[(x, y)] 变成 (batch_x, batch_y)）
    #     4) num_workers=0 时就在主进程里顺序做；>0 时交给子进程做
    #
    # DataLoader 是「可重复迭代」的：同一份数据集可以反复 for，每次都会重新采样。
    # 但注意——DataLoader 本身不是迭代器，`next(dataloader)` 会报错，要先用 `iter(dataloader)`。

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,          # 训练集必须打乱，否则同类样本扎堆，梯度方向有偏
        num_workers=0,         # Windows 上先设 0，原因见第 5 节
        drop_last=False,       # 保留最后不满一个 batch 的残块
    )

    print(f"len(train_loader) = {len(train_loader)}  ← DataLoader 的「长度」= batch 个数")
    print("\n逐个 batch 遍历：")
    for step, (X_batch, y_batch) in enumerate(train_loader):
        print(f"  batch {step}: X shape={tuple(X_batch.shape)}, dtype={X_batch.dtype} | "
              f"y shape={tuple(y_batch.shape)}, dtype={y_batch.dtype}")
    print()

    # 用 iter() + next() 手动取一个 batch（调试常用）
    it = iter(train_loader)
    X_one, y_one = next(it)
    print(f"iter(dataloader) + next() 手动取一个 batch：X {tuple(X_one.shape)}, y {tuple(y_one.shape)}")
    print(f"  这个 batch 的标签 = {y_one.tolist()}")
    print()


    # ===========================================================================
    # 【3】drop_last 对「最后一个不满的 batch」的影响
    # ===========================================================================
    print("-" * 78)
    print("【3】drop_last=False/True：100 条样本 / batch_size=16")
    print("-" * 78)

    # ---- 原理 ----
    # 100 / 16 = 6.25，前 6 个 batch 各 16 条（共 96 条），最后剩 4 条。
    #     drop_last=False（默认）：保留那 4 条 → 一共 7 个 batch，最后一个是 4。
    #     drop_last=True          ：丢掉那 4 条 → 一共 6 个 batch，每批都是 16。
    # 什么时候用 True？
    #     · 模型里有 BatchNorm 时：一个 batch 只有 1~2 条样本，统计量极不稳定，
    #       会产生很脏的梯度，训练反而变差，所以常见做法是 drop_last=True。
    #     · 需要固定形状（如某些 CUDA kernel / 编译图）时。
    # 什么时候用 False？
    #     · 数据本来就少，舍不得丢（比如只有 100 条样本的小实验）。
    #     · 验证/测试集**必须**用 False，否则会漏评样本、指标算不准。
    n_total, bs = len(dataset), 16
    n_full = n_total // bs                                 # 完整 batch 个数
    n_left = n_total % bs                                  # 最后残块的样本数
    print(f"样本总数 {n_total}，batch_size {bs} → 整除部分 {n_full} 个满 batch，余下 {n_left} 条")

    loader_keep = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)
    loader_drop = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True, num_workers=0)

    shapes_keep = [tuple(X.shape) for X, _ in loader_keep]
    shapes_drop = [tuple(X.shape) for X, _ in loader_drop]
    print(f"\ndrop_last=False：len(dataloader)={len(loader_keep)}，各 batch 形状 = {shapes_keep}")
    print(f"drop_last=True ：len(dataloader)={len(loader_drop)}，各 batch 形状 = {shapes_drop}")

    # 用 math.ceil 说明 len(dataloader) 的通用公式
    ceil_formula = math.ceil(n_total / bs)
    floor_formula = (n_total - 1) // bs + 1      # 纯整数写法，效果同 ceil
    print(f"\n通用公式（drop_last=False）：len(dataloader) = ceil(n / batch_size) = ceil({n_total}/{bs}) = {ceil_formula}")
    print(f"  等价的纯整数写法 (n - 1) // batch_size + 1 = {floor_formula}  两者一致：{ceil_formula == floor_formula == len(loader_keep)}")
    print(f"通用公式（drop_last=True ）：len(dataloader) = floor(n / batch_size) = {n_full}  一致：{n_full == len(loader_drop)}")

    # 验证「丢掉的到底是什么」：drop_last=True 比 False 少了哪几条样本
    kept_indices = set()
    for X_b, _ in loader_drop:
        for row in X_b:
            kept_indices.add(tuple(row.tolist()))
    all_indices = set(tuple(dataset.X[i].tolist()) for i in range(n_total))
    dropped = all_indices - kept_indices
    print(f"\ndrop_last=True 丢掉的样本数 = {len(dropped)}（正好是被切掉的那个残块）")
    print(f"drop_last=True 时训练用到的样本数 = {len(kept_indices)} / {n_total}，即丢掉了 {len(dropped) / n_total * 100:.1f}% 的数据")
    print()


    # ===========================================================================
    # 【4】shuffle 对取数顺序的影响
    # ===========================================================================
    print("-" * 78)
    print("【4】shuffle=True/False：取数顺序对比（看前 8 个样本的标签）")
    print("-" * 78)

    # ---- 原理 ----
    # shuffle=True 时 DataLoader 用 RandomSampler：每个 epoch 生成一个随机的下标排列。
    #   为什么训练要打乱？梯度下降假设每批样本是「总体的无偏估计」。
    #   若数据按类别排好序（真实数据集经常如此：前 1000 张全是猫），
    #   不打乱就会连续多个 batch 全是同一类，模型会来回震荡、收敛极慢。
    # shuffle=False 时用 SequentialSampler：严格按下标 0,1,2,... 取，便于复现、便于对齐调试。
    # 注意：整个 epoch 里所有样本都会被取到一次且仅一次（RandomSampler 是无放回抽样）。
    #       所以 shuffle 只是改变顺序，不改变数据总量。
    loader_no_shuffle = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    loader_shuffle = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    first_no = next(iter(loader_no_shuffle))[1]
    first_sh = next(iter(loader_shuffle))[1]
    print(f"shuffle=False 的第 1 个 batch 标签 = {first_no.tolist()}")
    print(f"shuffle=True  的第 1 个 batch 标签 = {first_sh.tolist()}")
    print(f"两者是否相同：{torch.equal(first_no, first_sh)}  ← 顺序变了")

    # 再看「每个样本被访问到的顺序」
    torch.manual_seed(0)
    order_no = []
    for _, y_b in loader_no_shuffle:
        order_no.extend(y_b.tolist())
    torch.manual_seed(0)
    order_sh = []
    for _, y_b in loader_shuffle:
        order_sh.extend(y_b.tolist())
    print(f"\nshuffle=False 全部 100 条标签的前 16 个：{order_no[:16]}")
    print(f"shuffle=True  全部 100 条标签的前 16 个：{order_sh[:16]}")
    print(f"两种方式取到的样本总数：{len(order_no)} vs {len(order_sh)}（都是 100，没丢也没重复）")
    print(f"两种方式标签计数是否一致（证明只是换顺序）："
          f"{sorted(order_no) == sorted(order_sh)}")

    # 小实验：打乱能让「每个 batch 的正类比例」更接近总体比例
    pos_ratio_all = float(dataset.y.float().mean())
    ratios_no = [float(y_b.float().mean()) for _, y_b in DataLoader(dataset, batch_size=16, shuffle=False)]
    torch.manual_seed(1)
    ratios_sh = [float(y_b.float().mean()) for _, y_b in DataLoader(dataset, batch_size=16, shuffle=True)]
    print(f"\n总体正类比例 = {pos_ratio_all:.3f}")
    print(f"shuffle=False 各 batch 正类比例 = {[round(v, 3) for v in ratios_no]}")
    print(f"shuffle=True  各 batch 正类比例 = {[round(v, 3) for v in ratios_sh]}")
    print(f"  → 波动幅度（标准差）：不打乱 {np.std(ratios_no):.4f}，打乱后 {np.std(ratios_sh):.4f}")
    print("  → 打乱的目的是让每个 batch 都尽量接近总体分布，梯度方向更稳定。")
    print()


    # ===========================================================================
    # 【5】num_workers / pin_memory / __main__ 保护
    # ===========================================================================
    print("-" * 78)
    print("【5】num_workers 与 Windows 的 spawn 陷阱")
    print("-" * 78)

    # ---- 原理 ----
    # num_workers=0（默认）：整个数据加载在主进程里同步做，最简单、最好调试。
    # num_workers>0：DataLoader 会 fork / spawn 出 N 个子进程并行读数据，
    #                主进程只负责收集结果，从而把「读磁盘 + 解码图片 + 数据增强」
    #                这些 CPU 密集工作与 GPU 计算重叠起来（GPU 才不会饿着）。
    #
    # ★ Windows 的关键差异：
    #   Linux 用 fork：子进程直接复制父进程内存，所以在任何地方建 DataLoader 都没问题。
    #   Windows 没有 fork，只能 spawn：子进程会**重新 import 主模块**并从头执行一遍。
    #   如果此时模块顶层又创建了 DataLoader，就会无限递归地创建子进程 → 报错/卡死。
    #   因此 Windows 上必须把「创建 DataLoader 和训练循环」放进
    #       if __name__ == "__main__": 保护块里，
    #   这样子进程重新 import 模块时不会重复执行训练代码。
    #
    # 本脚本为了「一行命令直接跑」把大部分代码写在模块顶层，所以正式演示统一用 num_workers=0；
    # 第 6 节会在 __main__ 保护下真正开 2 个子进程验证一次。

    print("num_workers 对比说明：")
    print("  num_workers=0 : 主进程同步加载。Windows 上最省心，调试友好，小数据集足够快。")
    print("  num_workers>0 : 子进程并行加载。Windows 用 spawn，必须配合 if __name__ == '__main__': 保护。")
    print("  经验值        : num_workers = min(4, os.cpu_count()) 起步，再按 IO 压力调整。")

    # pin_memory 说明：把 batch 放进「锁页内存（page-locked memory）」，
    # 这样 CPU→GPU 的 H2D 拷贝可以用 DMA 直接搬，不用先经过可分页内存，速度更快。
    # 但本机是 CPU 版 torch，没有 GPU 可拷，pin_memory=True 只会白白多一次拷贝开销，所以保持 False。
    print("\npin_memory 的用途：")
    print("  · True 时把 batch 放进锁页内存，加速 CPU→GPU 的数据拷贝（GPU 训练建议 True）。")
    print(f"  · 本机 CUDA 可用 = {torch.cuda.is_available()}，没有 GPU 可拷，所以设为 False，避免多余开销。")




    # ===========================================================================
    # 【6】变长序列 + 自定义 collate_fn（padding + mask）
    # ===========================================================================
    print("-" * 78)
    print("【6】自定义 collate_fn：把变长序列 padding 到同一长度（附 attention mask）")
    print("-" * 78)

    # ---- 原理 ----
    # torch 只能处理「矩形」张量，而 NLP/语音里的序列长度天然不等（句子的词数不一样）。
    # 默认的 collate_fn（default_collate）遇到不等长会直接报错，
    # 所以必须自己定义 collate_fn，把一批样本「补齐」成同一个长度：
    #     padding：用特殊 token（这里用 0）把短序列补到本 batch 的最长长度 max_len。
    #     为什么要 mask？因为补出来的 0 是「假的」，不能参与 attention / pooling / loss 计算，
    #     所以要额外产出一个 (batch, max_len) 的布尔 mask，True 表示「这是真实 token」，
    #     False 表示「这是补位」。损失计算时通常写 loss = (loss * mask).sum() / mask.sum()。
    # 关键点：max_len 是**按 batch 动态计算**的（不是全局最大长度），
    #         这样短句子的 batch 就不会被全局最长的句子拖累，省算力。





    seq_lengths = [5, 3, 8, 1, 6, 2]
    seq_dataset = VariableLengthSeqDataset(seq_lengths)
    print(f"造了 {len(seq_dataset)} 条长度不同的序列，长度分别为 {seq_lengths}")
    for i, L in enumerate(seq_lengths):
        print(f"  dataset[{i}] → 形状 {tuple(seq_dataset[i].shape)}, 内容前 4 个 = {seq_dataset[i][:4].tolist()}")

    # 先看看「不用 collate_fn」会怎样（默认 default_collate 无法堆叠不等长张量）
    try:
        _ = DataLoader(seq_dataset, batch_size=6, collate_fn=None).__iter__().__next__()
        print("\n默认 collate_fn 竟然成功了？不应该发生")
    except Exception as exc:
        print(f"\n用默认 collate_fn 打包变长序列：抛出 {type(exc).__name__}")
        print("  → 已捕获。原因：default_collate 要求同一位置上的张量形状完全一致，长度不同就堆不起来。")
        print("     解决办法就是自定义 collate_fn 做 padding。")

    # 用自定义 collate_fn
    seq_loader = DataLoader(seq_dataset, batch_size=3, shuffle=False, collate_fn=pad_collate_fn)
    print("\n使用 pad_collate_fn（batch_size=3）：")
    for i, (padded, mask) in enumerate(seq_loader):
        print(f"  batch {i}: padded shape={tuple(padded.shape)}, mask shape={tuple(mask.shape)}")
        print(f"    padded = {padded.tolist()}")
        print(f"    mask   = {[['真' if b else '补' for b in row] for row in mask.tolist()]}")
        real_tokens = int(mask.sum())
        total_slots = padded.numel()
        print(f"    本 batch 真实 token 数 = {real_tokens}/{total_slots}（padding 占比 {(1 - real_tokens / total_slots) * 100:.1f}%）")
    print()

    # 说明 padding 之后怎么用 mask 算 loss（用均值代替，避免引入额外复杂度）
    padded_demo, mask_demo = next(iter(seq_loader))
    # 假装模型对每个位置输出了一个分数（这里直接用 padded 的浮点值当分数）
    fake_logits = padded_demo.float()
    fake_loss_per_token = fake_logits ** 2                      # (batch, max_len) 逐 token 损失
    masked_loss = (fake_loss_per_token * mask_demo).sum() / mask_demo.sum()
    unmasked_loss = fake_loss_per_token.mean()
    print(f"带 mask 的损失 = {masked_loss.item():.4f}（分母只数真实 token，padding 不参与）")
    print(f"不带 mask 的损失 = {unmasked_loss.item():.4f}（补位的 0 也被算进去，分母被稀释，损失偏小）")
    print("  → 这就是必须产出 mask 的原因：不屏蔽 padding，损失和梯度都会被「假的 0」污染。")
    print()


    # ===========================================================================
    # 【7】TensorDataset：最省事的数据集
    # ===========================================================================
    print("-" * 78)
    print("【7】TensorDataset：把多个张量按第 0 维对齐后包成 Dataset")
    print("-" * 78)

    # ---- 原理 ----
    # 当数据已经在内存里、就是几个张量时，不必手写 Dataset 类，
    # 直接用 TensorDataset(X, y) 即可：它内部按第 0 维做索引，
    # __getitem__(i) 返回 (X[i], y[i], ...) —— 也就是「第 i 行」的元组。
    # 要求：所有传入张量的第 0 维长度必须相同（否则索引越界）。
    X_all = torch.randn(64, 5)
    y_all = torch.randint(0, 3, (64,))               # 3 分类标签

    tensor_ds = TensorDataset(X_all, y_all)
    print(f"TensorDataset(X{tuple(X_all.shape)}, y{tuple(y_all.shape)})")
    print(f"  len(tensor_ds) = {len(tensor_ds)}")
    print(f"  tensor_ds[0]   = (X[0] shape {tuple(tensor_ds[0][0].shape)}, y[0] = {tensor_ds[0][1].item()})")

    tensor_loader = DataLoader(tensor_ds, batch_size=20, shuffle=True, num_workers=0)
    print(f"  DataLoader(batch_size=20) → len = {len(tensor_loader)}（ceil(64/20) = {math.ceil(64 / 20)}）")
    for i, (xb, yb) in enumerate(tensor_loader):
        print(f"    batch {i}: X{tuple(xb.shape)} y{tuple(yb.shape)}")
    print("  → TensorDataset + DataLoader 两行就能跑通，适合快速实验；复杂读取再换自定义 Dataset。")
    print()


    # ===========================================================================
    # 【8】len(dataset) 与 len(dataloader) 的关系
    # ===========================================================================
    print("-" * 78)
    print("【8】len(dataset) 与 len(dataloader) 的关系")
    print("-" * 78)

    # ---- 原理 ----
    #   len(dataset)    = 样本总数 n
    #   len(dataloader) = batch 个数：
    #         drop_last=False → ceil(n / batch_size) = (n - 1) // batch_size + 1
    #         drop_last=True  → floor(n / batch_size) = n // batch_size
    # 一个 epoch 里，DataLoader 会被完整遍历一次，即模型会看到 n 条样本（drop_last=False 时）。
    # 训练时的总迭代次数 = len(dataloader) × epochs。
    print(f"{'n':>5} {'batch_size':>11} {'drop_last':>10} {'len(dataloader)':>16} {'实际喂给模型的样本数':>22}")
    rows = []
    for n in (100, 64, 17):
        for bs_i in (16, 8):
            for drop in (False, True):
                ds_i = TensorDataset(torch.arange(n).float().unsqueeze(1), torch.arange(n))
                dl = DataLoader(ds_i, batch_size=bs_i, drop_last=drop, num_workers=0)
                seen = sum(xb.shape[0] for xb, _ in dl)
                rows.append((n, bs_i, drop, len(dl), seen))
                print(f"{n:>5} {bs_i:>11} {str(drop):>10} {len(dl):>16} {seen:>22}")

    print("\n规律：drop_last=True 时 len(dataloader) = n // batch_size，实际样本数可能少于 n（有浪费）；")
    print("      drop_last=False 时 len(dataloader) = ceil(n / batch_size)，样本一条不漏。")
    print(f"训练总迭代次数 = len(dataloader) × epochs，例如 7 × 5 = {7 * 5} 次参数更新。")
    print()


    # ===========================================================================
    # 【9】训练 / 验证集切分：random_split 与 Subset
    # ===========================================================================
    print("-" * 78)
    print("【9】用 random_split / Subset 切分训练集与验证集")
    print("-" * 78)

    # ---- 原理 ----
    # 为什么必须切分？模型在训练集上「背下来」不代表会做新题（过拟合）。
    # 必须留一部分数据只看不训，用来估计真实泛化能力。
    # 两种常用做法：
    #   random_split(dataset, [n_train, n_val])  —— 随机切分，返回两个 Subset（共享底层数据，不复制）
    #   Subset(dataset, indices)                 —— 用**显式下标**切分，适合「按时间切分」或分层抽样
    # 注意：两者都是「视图」，不会拷贝底层张量，所以内存开销几乎为零；
    #       但也不能通过 Subset 修改原始数据（要改就改底层 dataset）。
    generator_split = torch.Generator().manual_seed(42)
    n_train_split = int(len(dataset) * 0.8)
    n_val_split = len(dataset) - n_train_split
    train_subset, val_subset = random_split(dataset, [n_train_split, n_val_split], generator=generator_split)

    print(f"random_split(dataset, [{n_train_split}, {n_val_split}])")
    print(f"  训练子集长度 = {len(train_subset)}，验证子集长度 = {len(val_subset)}，合计 = {len(train_subset) + len(val_subset)}")
    print(f"  类型 = {type(train_subset).__name__}（Subset 是视图，不复制底层数据）")

    # 检查两个子集是否完全不重叠（切分正确性验证）
    train_indices = set(train_subset.indices)
    val_indices = set(val_subset.indices)
    print(f"  训练下标数 = {len(train_indices)}，验证下标数 = {len(val_indices)}，交集 = {len(train_indices & val_indices)}（应为 0）")

    train_loader_split = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader_split = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
    print(f"  train_loader: len={len(train_loader_split)}，batch 形状 = {[tuple(x.shape) for x, _ in train_loader_split]}")
    print(f"  val_loader  : len={len(val_loader_split)}，batch 形状 = {[tuple(x.shape) for x, _ in val_loader_split]}（验证集不 shuffle）")

    # 显式 Subset：比如只取前 30 条做「快速冒烟测试」，或用 numpy 做分层抽样后传下标
    smoke_indices = list(range(30))
    smoke_subset = Subset(dataset, smoke_indices)
    smoke_loader = DataLoader(smoke_subset, batch_size=16, num_workers=0)
    print(f"\nSubset(dataset, range(30)) → len={len(smoke_subset)}，loader 长度={len(smoke_loader)}"
          f"（{math.ceil(30 / 16)} 个 batch）")
    print("  → 冒烟测试时只跑几十条样本，几秒内就能发现「形状写错 / 设备不一致」这类低级错误。")
    print()


    # ===========================================================================
    # 【10】手写 MyDataLoader：只用索引 + 列表推导实现 shuffle + 分批
    # ===========================================================================
    print("-" * 78)
    print("【10】手写 MyDataLoader vs 官方 DataLoader")
    print("-" * 78)

    # ---- 原理 ----
    # DataLoader 看起来很神奇，但核心逻辑就是三行：
    #     1) 生成下标列表 range(n)
    #     2) shuffle=True 就打乱这个下标列表
    #     3) 按 batch_size 切片，对每一片做 [dataset[i] for i in 切片]，
    #        再把结果「按位置转置」成 (batch_X, batch_y)
    # 自己写一遍能彻底破除「黑盒感」，也能明白为什么 DataLoader 慢的时候要往哪查
    # （慢通常慢在 __getitem__ 里的 IO 和解码，而不是分批逻辑本身）。




    my_loader = MyDataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, seed=42)
    print(f"手写 MyDataLoader：len = {len(my_loader)}（官方 drop_last=False 时也是 {len(loader_keep)}）")
    for i, (xb, yb) in enumerate(my_loader):
        print(f"  batch {i}: X{tuple(xb.shape)} y{tuple(yb.shape)}")
    print()

    # ---- 与官方 DataLoader 逐项对比 ----
    print("对比 1：batch 个数与形状（都 shuffle=False）")
    my_ns = MyDataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    official_ns = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    my_shapes = [tuple(x.shape) for x, _ in my_ns]
    of_shapes = [tuple(x.shape) for x, _ in official_ns]
    print(f"  手写   : {my_shapes}")
    print(f"  官方   : {of_shapes}")
    print(f"  完全一致：{my_shapes == of_shapes}")

    print("\n对比 2：数据内容（shuffle=False 时顺序应完全相同）")
    my_X = torch.cat([x for x, _ in MyDataLoader(dataset, 16, shuffle=False)])
    of_X = torch.cat([x for x, _ in DataLoader(dataset, 16, shuffle=False, num_workers=0)])
    print(f"  拼接后的 X 形状 {tuple(my_X.shape)} vs {tuple(of_X.shape)}，内容一致：{torch.equal(my_X, of_X)}")

    print("\n对比 3：shuffle 效果（同一份数据、同样的打乱种子）")
    my_sh = MyDataLoader(dataset, 16, shuffle=True, seed=42)
    official_sh = DataLoader(dataset, 16, shuffle=True, num_workers=0,
                             generator=torch.Generator().manual_seed(42))
    my_order = torch.cat([y for _, y in my_sh])
    of_order = torch.cat([y for _, y in official_sh])
    print(f"  手写的第 1 个 batch 标签   = {my_order[:8].tolist()}")
    print(f"  官方的第 1 个 batch 标签   = {of_order[:8].tolist()}")
    print(f"  两者标签多重集是否一致（证明打乱不丢样本）：{sorted(my_order.tolist()) == sorted(of_order.tolist())}")
    print(f"  是否一个都没漏：手写 {len(my_order)} 条，官方 {len(of_order)} 条，原始 {len(dataset)} 条")
    print("\n  说明：手写用 random.Random(42)，官方用 torch.Generator(42)，两者随机数流不同，")
    print("        所以「打乱后的具体顺序」不要求相同；但都必须是原数据的无重复排列。")

    print("\n对比 4：drop_last 行为")
    my_drop = [tuple(x.shape) for x, _ in MyDataLoader(dataset, 16, shuffle=False, drop_last=True)]
    of_drop = [tuple(x.shape) for x, _ in DataLoader(dataset, 16, shuffle=False, drop_last=True, num_workers=0)]
    print(f"  手写 drop_last=True : {my_drop}")
    print(f"  官方 drop_last=True : {of_drop}")
    print(f"  完全一致：{my_drop == of_drop}")
    print("\n结论：官方 DataLoader 多出来的东西是「多进程 / pin_memory / sampler 体系 / collate 泛化」，")
    print("      而分批与打乱的核心逻辑就这么简单。理解了它，调参和排查数据问题就不再是黑盒。")
    print()


    # ===========================================================================
    # 【11】绘图：batch 数对比 + padding mask + 两种加载器覆盖验证
    # ===========================================================================
    print("-" * 78)
    print("【11】绘图：drop_last 影响 / padding mask / 加载器覆盖验证")
    print("-" * 78)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    fig.suptitle("Dataset 与 DataLoader：batch 划分、padding mask 与手写加载器验证", fontsize=15, fontweight="bold")

    # ---- 子图 1：100 条样本在不同设置下每个 batch 的样本数 ----
    ax = axes[0]
    counts_keep = [x.shape[0] for x, _ in loader_keep]
    counts_drop = [x.shape[0] for x, _ in loader_drop]
    xpos_keep = np.arange(len(counts_keep))
    ax.bar(xpos_keep, counts_keep, 0.6, color="#4c72b0", edgecolor="white", label=f"drop_last=False（{len(counts_keep)} 个 batch）")
    ax.bar(np.arange(len(counts_drop)), counts_drop, 0.6, color="#55a868", edgecolor="white", label=f"drop_last=True（{len(counts_drop)} 个 batch）")
    for xi, c in zip(xpos_keep, counts_keep):
        ax.text(xi, c + 0.3, str(c), ha="center", fontsize=9, color="#4c72b0")
    for xi, c in zip(np.arange(len(counts_drop)), counts_drop):
        ax.text(xi + 0.02, c + 0.3, str(c), ha="center", fontsize=9, color="#55a868")
    ax.set_xticks(xpos_keep, [f"#{i}" for i in xpos_keep], fontsize=8)
    ax.set_xlabel("batch 序号", fontsize=10)
    ax.set_ylabel("该 batch 的样本数", fontsize=10)
    ax.set_ylim(0, 20)
    ax.set_title("① 100 条样本 / batch_size=16：6 满 + 1 残(=4)", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    ax.annotate("最后一批只剩 4 条\n（drop_last 决定去留）", xy=(6, 4), xytext=(3.6, 13),
                fontsize=8.5, color="#c00000",
                arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.4))

    # ---- 子图 2：padding mask 热力图 ----
    ax = axes[1]
    mask_show = mask_demo.numpy().astype(int)
    im = ax.imshow(mask_show, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_title("② collate_fn 产出的 attention mask", fontsize=11)
    ax.set_xlabel("序列位置（padding 到 max_len）", fontsize=10)
    ax.set_ylabel("batch 内样本序号", fontsize=10)
    ax.set_xticks(range(mask_show.shape[1]))
    ax.set_yticks(range(mask_show.shape[0]), [f"样本{i}\n(len={L})" for i, L in enumerate(seq_lengths[:mask_show.shape[0]])], fontsize=8.5)
    for i in range(mask_show.shape[0]):
        for j in range(mask_show.shape[1]):
            ax.text(j, i, "真" if mask_show[i, j] else "补", ha="center", va="center",
                    fontsize=8, color="white" if mask_show[i, j] else "#888888")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1]).set_label("1=真实 token  0=padding", fontsize=8.5)

    # ---- 子图 3：手写/官方加载器的数据覆盖验证（标签直方图） ----
    ax = axes[2]
    all_labels_official = torch.cat([y for _, y in DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                                                              generator=torch.Generator().manual_seed(3))])
    all_labels_mine = torch.cat([y for _, y in MyDataLoader(dataset, batch_size=16, shuffle=True, seed=3)])
    counts_official = [int((all_labels_official == c).sum()) for c in (0, 1)]
    counts_mine = [int((all_labels_mine == c).sum()) for c in (0, 1)]
    pos_bar = np.arange(2)
    ax.bar(pos_bar - 0.18, counts_official, 0.34, color="#4c72b0", edgecolor="white", label="官方 DataLoader")
    ax.bar(pos_bar + 0.18, counts_mine, 0.34, color="#c44e52", edgecolor="white", label="手写 MyDataLoader")
    for xi, c in zip(pos_bar - 0.18, counts_official):
        ax.text(xi, c + 0.6, str(c), ha="center", fontsize=9, color="#4c72b0")
    for xi, c in zip(pos_bar + 0.18, counts_mine):
        ax.text(xi, c + 0.6, str(c), ha="center", fontsize=9, color="#c44e52")
    ax.set_xticks(pos_bar, ["类别 0", "类别 1"], fontsize=10)
    ax.set_ylabel("取到的样本数", fontsize=10)
    ax.set_ylim(0, max(counts_official + counts_mine) * 1.3)
    ax.set_title("③ 一个 epoch 的类别覆盖（两者都取满全量）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.text(0.5, 0.04, "手写加载器只差多进程/采样器体系，分批逻辑等价",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#404040")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = OUTPUT_DIR / "03_数据集与DataLoader_batch与padding.png"
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"图已保存：{save_path}")
    print(f"文件存在：{save_path.exists()}，大小：{save_path.stat().st_size / 1024:.1f} KB")
    print()


    # ===========================================================================
    # 【12】多进程加载 + 真实训练循环
    # ===========================================================================
    print("-" * 78)
    print("【12】num_workers>0 的多进程加载，并真的跑两轮训练循环")
    print("-" * 78)

    print("为什么不要写在模块顶层？")
    print("  · Windows 用 spawn 启动子进程：子进程会**重新 import 本模块**（此时模块名是 __mp_main__）。")
    print("  · 所以：会被 pickle 传给子进程的类（如 CustomDataset）必须定义在**模块级**；")
    print("  · 而演示 / 训练代码必须放进 main() 并加 if __name__ == '__main__': 保护，")
    print("    这样子进程重新导入模块时只拿到「类定义」，不会把整个演示再跑一遍。")
    print("  · 本脚本就是这样组织的：上面的 CustomDataset / VariableLengthSeqDataset / MyDataLoader")
    print("    都在模块级，下面 main() 里才是会被执行的演示代码。")

    # 下面用 num_workers=0 和 2 各跑一遍同样的训练，先确认「数据完全一致」，再看耗时差异。
    bs_train, n_train_demo = 32, 96
    train_ds = CustomDataset(n_samples=n_train_demo, n_features=8)
    val_ds = CustomDataset(n_samples=32, n_features=8, seed=7)

    loss_func = torch.nn.CrossEntropyLoss()

    print("\n用一个两层小网络真的跑 2 轮训练（数据 96 条，batch_size=32）：")
    # 只开 1 个子进程就足够展示 spawn 行为；开 2 个能再快一点，但在本机 spawn 开销下整体更慢，
    # 权衡后取 1（教学效果好且不让脚本耗时翻倍，你可以把它改成 2 或 4 自己实测一下）。
    for nw in (0, 1):
        # 换 num_workers 就要重建 DataLoader；注意 Windows 上 num_workers>0 会 spawn 子进程
        loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, num_workers=nw, drop_last=False)
        net = _make_model()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
        t_start = time.perf_counter()
        losses = []
        for epoch in range(2):
            net.train()                              # 训练模式（Dropout/BatchNorm 行为不同）
            epoch_loss, n_seen = 0.0, 0
            for X_batch, y_batch in loader:          # ← 多进程时这里由子进程并行读数据
                optimizer.zero_grad()                # 1) 清空上一轮梯度（grad 是累加的！）
                outputs = net(X_batch)               # 2) 前向
                loss = loss_func(outputs, y_batch)   # 3) 算损失
                loss.backward()                      # 4) 反向传播
                optimizer.step()                     # 5) 更新参数
                epoch_loss += loss.item() * X_batch.shape[0]
                n_seen += X_batch.shape[0]
            losses.append(epoch_loss / n_seen)
        elapsed = time.perf_counter() - t_start

        # 顺带验证「多进程读到的数据」和「单进程读到的数据」逐元素一致（正确性 > 速度）
        labels_multiproc, same_as_single = _compare_worker_reads(train_ds, bs_train, nw)

        # 评估：换 eval() + no_grad()（验证集 shuffle=False、drop_last=False，一条都不漏）
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0):
                pred = net(X_batch).argmax(dim=1)
                correct += int((pred == y_batch).sum())
                total += y_batch.shape[0]
        print(f"  num_workers={nw}：batch 数={len(loader)}，各轮平均 loss={[round(v, 4) for v in losses]}，"
              f"验证准确率={correct / total:.3f}，耗时={elapsed:.2f} 秒，"
              f"数据与单进程一致={same_as_single}")

    print("\n说明：")
    print("  · 两种 num_workers 的 loss 走势和准确率在同一量级，说明并行加载不改变语义。")
    print("  · 本脚本数据是内存里的随机张量，没有磁盘 IO，所以 num_workers=1 反而更慢")
    print("    （Windows 上 spawn 一次要几百毫秒到几秒，只有数据量大到读盘/解码成为瓶颈时才划算）。")
    print("  · 所以调 num_workers 的经验法则是：先 0 保证正确，再逐步加，用实测时间决定，别盲目调大。")
    print()


    # ===========================================================================
    # 小结
    # ===========================================================================
    print("=" * 78)
    print("小结")
    print("=" * 78)
    print("· Dataset 管「单条样本怎么读」：必须实现 __init__ / __len__ / __getitem__")
    print("· DataLoader 管「怎么组成 batch」：batch_size / shuffle / num_workers / collate_fn / pin_memory")
    print("· 100 条样本 batch_size=16：drop_last=False → 7 个 batch（6×16 + 4）；True → 6 个 batch")
    print("· shuffle=True 让每个 batch 的类别分布接近总体，训练更稳；验证集用 shuffle=False")
    print("· Windows 上 num_workers>0 需要 __main__ 保护，且要被 pickle 的类必须定义在模块级")
    print("· 变长序列必须自定义 collate_fn 做 padding，并同时产出 mask，否则 padding 会污染 loss")
    print("· TensorDataset(X, y) 按第 0 维对齐，两行代码就能跑通训练循环")
    print("· len(dataloader) = ceil(n/batch_size)（drop_last=False）或 n//batch_size（drop_last=True）")
    print("· random_split / Subset 切分训练集与验证集，都是零拷贝的视图")
    print("· 手写 MyDataLoader = 打乱下标 + 按 batch_size 切片 + 列表推导 + zip(*samples) 转置")
    print("· 训练循环五行：zero_grad → forward → loss → backward → step")
    print("=" * 78)
    print("脚本执行完毕，退出码 0。")


if __name__ == "__main__":
    main()
