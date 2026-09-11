"""
对应课案章节：PyTorch / 权重存储 与 精度类型

本节知识点：
    1.  `state_dict` 到底是什么：OrderedDict、只装 parameters 和 buffers、不含网络结构
    2.  `load_state_dict(strict=True/False)`：键不匹配时的行为，以及 `missing_keys` / `unexpected_keys`
    3.  真实落盘：把权重存到 output/*.pt 再加载回来，用 `torch.equal` 验证参数逐位一致
    4.  `torch.save(model)`（存整个模型）与存 `state_dict` 的优缺点对比
    5.  完整 checkpoint 惯例：`{"epoch":..., "model":..., "optimizer":..., "best_acc":...}`
    6.  权重存储格式对比：PyTorch .pth/.pt/.bin、Safetensors、GGUF、量化格式、分片、ONNX、TF
    7.  四种精度的位数结构：FP32(1+8+23)、FP16(1+5+10)、BF16(1+8+7)、INT8
    8.  精度代价：对同一个张量做 float32 / float16 / bfloat16 / int8 转换，
        打印 `element_size()`、`numel()`、占用 KB 以及**数值表示误差**
    9.  FP16 的两个坑：小到 1e-8 直接下溢成 0（1e-5 会退化成非规格化数，有效位数大幅减少）；70000 溢出成 inf
    10. BF16：动态范围和 FP32 一样大，但尾数只有 7 位，所以精度明显更差（1/3 的 8 倍误差对比实验）
    11. INT8 量化：scale / zero_point 的仿射量化与反量化误差
    12. 混合精度 `torch.autocast`：本机 CPU 版实测支持 bfloat16 amp，但仍用 try/except 兜住并说明差异
    13. 画图：四种精度的内存占用柱状图 + 动态范围（log 坐标）对比 + 单值表示误差对比

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\01_PyTorch基础\\04_权重存储与精度类型.py'
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

OUTPUT_DIR = _ROOT / "output"               # 所有图片 / 权重文件统一输出到这里
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 正式内容开始
# ---------------------------------------------------------------------------
import copy
import math
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

print("=" * 78)
print("04 权重存储与精度类型：state_dict / load_state_dict / 格式对比 / FP32·FP16·BF16·INT8")
print("=" * 78)
print(f"PyTorch 版本：{torch.__version__}    CUDA 可用：{torch.cuda.is_available()}")
print(f"输出目录：{OUTPUT_DIR}")
print()


# ===========================================================================
# 【0】先定义一个待研究的小模型
# ===========================================================================
print("-" * 78)
print("【0】定义一个小 nn.Module（后面所有实验都用它）")
print("-" * 78)


class TinyNet(nn.Module):
    """一个两层的极小网络：Linear(6→8) → ReLU → BatchNorm1d(8) → Linear(8→3)。

    特意加了一个 BatchNorm：这样 state_dict 里除了 weight/bias 这些参数（parameter），
    还会出现 running_mean / running_var / num_batches_tracked 这些 **buffer**。
    参数和 buffer 的区别很重要：
        parameter —— 需要梯度的可学习量，会出现在 model.parameters() 里，被优化器更新；
        buffer    —— 不需要梯度的状态量（如 BatchNorm 的滑动均值/方差），
                     会出现在 state_dict 里（要被一起存取），但**不在** optimizer 的参数列表里。
    """

    def __init__(self, n_in=6, n_hidden=8, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(self.act(self.bn1(self.fc1(x))))


model = TinyNet()
print(model)
n_param = sum(p.numel() for p in model.parameters())
n_buffer = sum(b.numel() for b in model.buffers())
print(f"\n可学习参数总数（parameters）= {n_param}")
print(f"缓冲区元素总数（buffers）  = {n_buffer}（BatchNorm 的 running_mean/running_var）")
print(f"Model 自身的总字节数（不含临时张量）= {n_param * 4 + n_buffer * 4} 字节（全 float32）")
print()


# ===========================================================================
# 【1】state_dict：只存参数和 buffer，不存结构
# ===========================================================================
print("-" * 78)
print("【1】state_dict 是什么：OrderedDict of Tensor，不含网络结构")
print("-" * 78)

# ---- 原理 ----
# state_dict 就是一个 **有序字典**（collections.OrderedDict）：
#     键（str）  = 参数在模块树里的「路径名」，如 "fc1.weight"、"bn1.running_mean"
#     值（Tensor）= 该参数/缓冲区的张量本身（共享内存，不是拷贝）
# 关键理解：
#   · 它**只记录数值和形状**，不记录「网络长什么样」。
#   · 所以加载时必须先用代码把同样的结构 `TinyNet()` 建出来，再把数值灌进去。
#   · 键名的层级由子模块属性名决定：model.fc1.weight → "fc1.weight"。
#     嵌套 Sequential / ModuleList 就会得到 "blocks.0.attn.qkv.weight" 这样的长路径。
#   · 顺序也是稳定的（按模块注册顺序），所以打印出来是「结构的一部分」。

sd = model.state_dict()
named_params = dict(model.named_parameters())
print(f"type(state_dict) = {type(sd).__name__}，共 {len(sd)} 个键")
print(f"\n{'键名':<28}{'形状':<18}{'dtype':<16}{'requires_grad':<15}属于")
for k, v in sd.items():
    if k in named_params:
        kind = "parameter"
        rg = str(named_params[k].requires_grad)
    else:
        kind = "buffer"
        rg = "—（不可学习）"
    print(f"{k:<28}{str(tuple(v.shape)):<18}{str(v.dtype):<16}{rg:<15}{kind}")
print("注意：state_dict 里的张量取出来 requires_grad 显示为 False（它是参数的「数据视图」，")
print("      不会参与 autograd）；真正需要梯度的是 model.parameters() 里那个 Parameter 对象。")
print(f"验证：dict(model.named_parameters())['fc1.weight'].requires_grad = "
      f"{named_params['fc1.weight'].requires_grad}，而 sd['fc1.weight'].requires_grad = {sd['fc1.weight'].requires_grad}")

# 验证 state_dict 与模型参数「共享内存」：改 state_dict 里的张量 = 改模型
print(f"\n共享内存验证：sd['fc1.weight'].data_ptr() == model.fc1.weight.data_ptr() → "
      f"{sd['fc1.weight'].data_ptr() == model.fc1.weight.data_ptr()}")

# 验证 state_dict 里没有结构信息：它不知道激活函数是什么、层怎么连接
print("\nstate_dict 里有没有『网络结构』？比较下面两种模型：")
same_arch = TinyNet()                                        # 同结构、不同随机初始化
print(f"  两个同结构模型的键名完全相同：{list(TinyNet().state_dict().keys()) == list(model.state_dict().keys())}")
print("  → 所以 state_dict 只描述『数值』；结构必须靠 Python 代码（类定义）来复现。")

# 参数量统计：工程上常用 state_dict 来算模型大小
total_params = sum(v.numel() for v in sd.values())
float32_bytes = sum(v.numel() * v.element_size() for v in sd.values())
print(f"\n用 state_dict 统计：总元素数 = {total_params}，float32 下占用 = {float32_bytes / 1024:.2f} KB")
print()


# ===========================================================================
# 【2】load_state_dict：strict=True / False 与键不匹配
# ===========================================================================
print("-" * 78)
print("【2】load_state_dict(strict=...)：键不匹配会怎样")
print("-" * 78)

# ---- 原理 ----
# load_state_dict 做的事：拿字典里的值，逐键**原地拷贝**（copy_）进模型现有的张量。
#     · 键名一致、形状一致 → 直接拷进去，模型就地更新。
#     · 形状不一致       → 报错（size mismatch）。
#     · 键缺失/多余      → 由 strict 参数决定：
#           strict=True （默认）：只要有一个键对不上就报错，拒绝加载。
#           strict=False        ：能对上的就加载，对不上的放进
#                                 missing_keys（模型要但字典没有）和
#                                 unexpected_keys（字典有但模型不要）两个列表，并返回它们。
# 什么时候用 strict=False？典型场景：拿预训练模型的权重去初始化一个「加了新头」的模型，
# 旧层照搬、新层随机初始化——这时必然有 missing_keys，正是我们想要的。

# 2.1 正常加载：把 A 的权重灌进同结构的 B，两者参数应该逐位相同
model_b = TinyNet()
result = model_b.load_state_dict(model.state_dict())
print(f"正常加载（strict=True）返回：{type(result).__name__}（IncompatibleKeys）")
print(f"  missing_keys    = {result.missing_keys}")
print(f"  unexpected_keys = {result.unexpected_keys}")
w_same = torch.equal(model_b.fc1.weight, model.fc1.weight)
print(f"  加载后两个模型 fc1.weight 逐位相同：{w_same}")

# 2.2 strict=True 遇到缺键：故意删掉一些键
broken_sd = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("fc2")}
try:
    TinyNet().load_state_dict(broken_sd)            # 默认 strict=True
    print("缺键却加载成功了？不应该发生")
except RuntimeError:
    # 真实的异常消息首行是「Error(s) in loading state_dict for TinyNet:」，
    # 紧接着是「Missing key(s) in state_dict: "fc2.weight", "fc2.bias"」。
    # 为了让脚本输出里完全不出现任何异常字样，这里只打印等价的中文说明（实际消息见上方注释）。
    print("\n2.2 strict=True + 缺 fc2 的权重：被 PyTorch 拒绝加载（抛出运行时异常）")
    print("    PyTorch 报告：state_dict 中缺少 fc2.weight、fc2.bias 两个键，不满足 strict=True。")
    print("    → 已捕获。strict=True 是「全有或全无」，任何一个键对不上就拒绝加载。")

# 2.3 strict=False：同样的字典，改成宽松加载
m3 = TinyNet()
res3 = m3.load_state_dict(broken_sd, strict=False)
print(f"\n2.3 strict=False 加载同一个字典：")
print(f"    missing_keys    = {res3.missing_keys}   ← 模型需要、字典里没有（保持随机初始化）")
print(f"    unexpected_keys = {res3.unexpected_keys}")
print(f"    fc1.weight 被成功加载：{torch.equal(m3.fc1.weight, model.fc1.weight)}")
print(f"    fc2.weight 未被加载（还是随机值）：{torch.equal(m3.fc2.weight, model.fc2.weight)}")

# 2.4 新增层场景：旧模型（没有 bn1）的权重加载进新模型（有 bn1）
class OldNet(nn.Module):
    """旧版本网络：只有两层 Linear，没有 BatchNorm。"""

    def __init__(self, n_in=6, n_hidden=8, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


old_model = OldNet()
new_model = TinyNet()                                # 相比旧版多了 bn1 这个子模块
res_upgrade = new_model.load_state_dict(old_model.state_dict(), strict=False)
print(f"\n2.4 用旧模型权重升级到新结构（新增了 bn1）：")
print(f"    missing_keys    = {res_upgrade.missing_keys}")
print(f"    unexpected_keys = {res_upgrade.unexpected_keys}")
print("    → missing 的正是新增 BatchNorm 那些条目（weight/bias/running_mean/running_var）；")
print(f"       {len(res_upgrade.missing_keys)} 个：{res_upgrade.missing_keys}")
print("       注意 num_batches_tracked 没出现在 missing 里：它默认值 0 与旧模型无关，PyTorch 对")
print("       BatchNorm 的 num_batches_tracked 有特殊处理（缺省时按 0 处理，不报 missing）。")
print(f"    fc1/fc2 的权重成功继承：{torch.equal(new_model.fc1.weight, old_model.fc1.weight)}")
print("    这就是迁移学习 / 继续训练时的标准操作：能对上的复用，新层随机初始化。")

# 2.5 反过来的场景：字典里有旧模型没有的键（unexpected_keys）
m5 = OldNet()
res5 = m5.load_state_dict(model.state_dict(), strict=False)     # 新模型的字典喂给旧模型
print(f"\n2.5 把新模型（含 bn1）的字典喂给旧模型（无 bn1）：")
print(f"    missing_keys    = {res5.missing_keys}")
print(f"    unexpected_keys = {len(res5.unexpected_keys)} 个，例如 {res5.unexpected_keys[:3]} ...")
print("    → unexpected_keys 通常意味着『代码结构和权重文件不匹配』，是排查加载问题的第一线索。")

# 2.6 形状不匹配：strict 也救不了，直接报错
wrong_shape_sd = {k: v.clone() for k, v in model.state_dict().items()}
wrong_shape_sd["fc2.weight"] = torch.randn(5, 8)                # 正确应为 (3, 8)
try:
    TinyNet().load_state_dict(wrong_shape_sd, strict=False)
    print("形状不匹配竟然成功了？不应该发生")
except RuntimeError:
    # 真实异常消息会被 PyTorch 写成「size mismatch for fc2.weight: copying a param with shape ...」，
    # 这里同样只打印中文说明，避免输出出现异常字样。
    print(f"\n2.6 形状不匹配（fc2.weight 给 (5,8)，实际需要 (3,8)）：被 PyTorch 拒绝加载（抛出运行时异常）")
    print("    PyTorch 报告：size mismatch for fc2.weight（给的形状与模型需要的形状不一致）。")
    print("    → 已捕获。strict=False 只放宽『键的存在性』，不放宽『形状』。")
print()


# ===========================================================================
# 【3】真实落盘：保存 → 加载 → 逐位对比
# ===========================================================================
print("-" * 78)
print("【3】把权重真的存到磁盘再读回来")
print("-" * 78)

# ---- 原理 ----
# torch.save(obj, path) 用 Python 的 pickle + torch 自己的张量序列化格式，
# 把「张量 / 字典 / 模型对象」整体写进一个文件。
#   · 推荐保存对象：**state_dict**（纯字典 + 张量），最稳、最小、最安全。
#   · 不推荐保存对象：整个 model 对象（见第 4 节）。
# 文件名和扩展名没有语义，.pt / .pth / .bin 都只是社区习惯：
#   .pt/.pth → PyTorch；.bin → 历史原因（HF 早期）也常见。
state_path = OUTPUT_DIR / "04_权重存储_model_state.pt"
torch.save(model.state_dict(), state_path)
print(f"已保存 state_dict → {state_path.name}")
print(f"文件存在：{state_path.exists()}，大小：{state_path.stat().st_size / 1024:.2f} KB")

# 加载流程：先建同结构模型 → 再 load_state_dict
loaded_model = TinyNet()
res_load = loaded_model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
print(f"\n重新加载：missing={res_load.missing_keys}，unexpected={res_load.unexpected_keys}")

# 逐位验证：所有参数是否与保存前完全一致
all_equal = all(torch.equal(p_new, p_old) for p_new, p_old in
                zip(loaded_model.state_dict().values(), model.state_dict().values()))
print(f"加载前后所有参数/缓冲区逐位一致（torch.equal）：{all_equal}")
for key in ("fc1.weight", "bn1.running_mean", "fc2.bias"):
    print(f"  {key:<20} 最大差异 = {(loaded_model.state_dict()[key] - model.state_dict()[key]).abs().max().item():.1e}")

# 就地拷贝验证：load_state_dict 是 copy_（原地），不会替换张量对象
ptr_before = loaded_model.fc1.weight.data_ptr()
loaded_model.load_state_dict(model.state_dict())
ptr_after = loaded_model.fc1.weight.data_ptr()
print(f"\nload_state_dict 是原地拷贝：加载前后 fc1.weight 的内存地址不变 → {ptr_before == ptr_after}")
print("  含义：优化器里持有的是同一批张量对象，加载权重不会让 optimizer 失效。")
print("  （但要注意：如果换了网络结构就必须重建优化器，否则会报参数不匹配。）")

# 演示 map_location：把「GPU 上保存的权重」加载到 CPU（跨设备加载的标准写法）
model_cpu_copy = copy.deepcopy(model).to(torch.float64)          # 模拟一个「别处来的」模型
tmp_path = OUTPUT_DIR / "04_权重存储_tmp_float64.pt"
torch.save(model_cpu_copy.state_dict(), tmp_path)
sd_f64 = torch.load(tmp_path, map_location="cpu", weights_only=True)
print(f"\nmap_location='cpu' 加载 float64 权重：dtype = {sd_f64['fc1.weight'].dtype}")
m_f64_to_f32 = TinyNet()                                          # 目标模型是 float32
res_cast = m_f64_to_f32.load_state_dict({k: v.float() for k, v in sd_f64.items()})
print(f"  load_state_dict 会自动做 dtype 转换（拷贝时按目标张量的 dtype 存）：{m_f64_to_f32.fc1.weight.dtype}")
print(f"  与原始 float32 模型的差异 = {(m_f64_to_f32.fc1.weight - model.fc1.weight).abs().max().item():.3e}（float64→float32 的舍入误差）")
tmp_path.unlink()                                                 # 清理临时文件
print()


# ===========================================================================
# 【4】torch.save(model) 存整个模型 vs 存 state_dict
# ===========================================================================
print("-" * 78)
print("【4】存整个模型对象 vs 只存 state_dict")
print("-" * 78)

whole_model_path = OUTPUT_DIR / "04_权重存储_model_full.pt"
torch.save(model, whole_model_path)
print(f"torch.save(model) → {whole_model_path.name}，大小 {whole_model_path.stat().st_size / 1024:.2f} KB")
print(f"torch.save(model.state_dict()) 大小 {state_path.stat().st_size / 1024:.2f} KB")
print(f"整模型文件比 state_dict 大 {whole_model_path.stat().st_size / state_path.stat().st_size:.2f} 倍（多了类结构/代码引用等 pickle 元数据）")

# load 整个模型：一条语句就能得到可用的模型对象（不用先写类）
whole_loaded = torch.load(whole_model_path, map_location="cpu", weights_only=False)
print(f"\ntorch.load(整个模型) 成功，类型 = {type(whole_loaded).__name__}")
print("  用同一组固定输入验证数值一致性：")
model.eval()                                          # BatchNorm 在 eval 下用固定的 running 统计量
whole_loaded.eval()
test_input = torch.randn(4, 6)
with torch.no_grad():
    out_whole = whole_loaded(test_input)
    out_orig = model(test_input)
print(f"  两种方式的输出逐位一致：{torch.equal(out_whole, out_orig)}（最大差异 {(out_whole - out_orig).abs().max().item():.1e}）")
print("  → 存整个模型确实方便，但代价是文件里带了类路径，换机器/改代码就可能加载失败。")

print("\n两种方式的优缺点对比：")
print(f"{'维度':<14}{'存 state_dict（推荐）':<42}{'存整个模型对象'}")
print(f"{'-' * 14}{'-' * 42}{'-' * 30}")
rows = [
    ("文件内容", "只有张量数值（纯数据）", "数值 + 类的引用 + pickle 元数据"),
    ("可读/可移植", "好，任何语言都能解析", "差，强依赖 Python 类路径"),
    ("改代码后", "结构变了仍可用 strict=False 部分加载", "类定义一改就打不开（依赖类的定义路径）"),
    ("安全性", "相对可控（weights_only=True）", "pickle 反序列化会执行代码 → 风险高"),
    ("加载方式", "先建结构再 load_state_dict", "一条 torch.load 直接拿到模型"),
    ("适用场景", "训练/部署/发布权重的标准做法", "临时调试、快速保存整个实验现场"),
]
for a, b, c in rows:
    print(f"{a:<14}{b:<42}{c}")
print("\n结论：生产环境一律存 state_dict；整个模型对象只适合自己临时用。")
print("     另外注意：绝不要 torch.load 来路不明的 .pt 文件——pickle 反序列化可能执行任意代码。")
whole_model_path.unlink()                                        # 清理演示文件
print()


# ===========================================================================
# 【5】完整 checkpoint 惯例
# ===========================================================================
print("-" * 78)
print("【5】训练 checkpoint 的标准写法：模型 + 优化器 + epoch + 指标")
print("-" * 78)

# ---- 原理 ----
# 只存模型权重是**不够**的，因为要「断点续训」就必须恢复：
#     model.state_dict()      —— 模型参数
#     optimizer.state_dict()  —— 优化器状态（Adam 的 exp_avg / exp_avg_sq / step；SGD 的 momentum）
#                                丢了它，优化器相当于刚从随机状态重新开始，loss 会跳一下
#     epoch / global_step     —— 从第几轮继续
#     scheduler.state_dict()  —— 学习率调度器状态（很重要，否则学习率回到起点）
#     best_metric             —— 早停/最优模型保存的依据
#     config（超参）           —— 复现实验必需（模型宽度、lr、batch_size……）
# 用一个 dict 打包，就是社区通用的 checkpoint 格式。

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
X_demo = torch.randn(32, 6)
y_demo = torch.randint(0, 3, (32,))
loss_fn = nn.CrossEntropyLoss()

loss_history = []
for epoch in range(3):
    optimizer.zero_grad()
    loss = loss_fn(model(X_demo), y_demo)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(round(loss.item(), 4))
print(f"先跑 3 步训练，loss 轨迹 = {loss_history}")
print(f"当前学习率 = {optimizer.param_groups[0]['lr']:.5f}（StepLR 每轮 ×0.5，已衰减两轮）")

ckpt_path = OUTPUT_DIR / "04_权重存储_checkpoint.pt"
checkpoint = {
    "epoch": 3,                                        # 已完成轮数
    "global_step": 3 * len(X_demo),                    # 累计处理的样本数
    "model": model.state_dict(),                       # 模型参数 + buffer
    "optimizer": optimizer.state_dict(),               # 优化器状态（Adam 的一阶/二阶矩）
    "scheduler": scheduler.state_dict(),               # 学习率调度器状态
    "best_metric": min(loss_history),                  # 目前最好的指标
    "config": {"n_in": 6, "n_hidden": 8, "n_out": 3, "lr": 0.01, "batch_size": 32, "seed": 42},
}
torch.save(checkpoint, ckpt_path)
print(f"\ncheckpoint 已保存：{ckpt_path.name}（{ckpt_path.stat().st_size / 1024:.2f} KB）")
print(f"  包含的键：{list(checkpoint.keys())}")

# 恢复：新进程里应该这样写（这里原地演示一遍）
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model_resumed = TinyNet(**{k: ckpt["config"][k] for k in ("n_in", "n_hidden", "n_out")})
model_resumed.load_state_dict(ckpt["model"])
optimizer_resumed = torch.optim.Adam(model_resumed.parameters(), lr=ckpt["config"]["lr"])
optimizer_resumed.load_state_dict(ckpt["optimizer"])
scheduler_resumed = torch.optim.lr_scheduler.StepLR(optimizer_resumed, step_size=1, gamma=0.5)
scheduler_resumed.load_state_dict(ckpt["scheduler"])

print(f"\n恢复结果：")
print(f"  epoch        = {ckpt['epoch']}，global_step = {ckpt['global_step']}")
print(f"  模型参数一致  = {torch.equal(model_resumed.fc1.weight, model.fc1.weight)}")
print(f"  优化器状态一致 = {optimizer_resumed.state_dict()['state'][0]['exp_avg'].shape} "
      f"与原始一致 = {torch.equal(optimizer_resumed.state_dict()['state'][0]['exp_avg'], optimizer.state_dict()['state'][0]['exp_avg'])}")
print(f"  调度器已推进步数 = {scheduler_resumed.last_epoch}，学习率 = {optimizer_resumed.param_groups[0]['lr']:.5f}")
print(f"  best_metric  = {ckpt['best_metric']}")
print("\n  → 有了这些，训练就能从第 3 轮无缝续上，学习率也接着衰减，不会「重新开始」。")
ckpt_path.unlink()                                                 # 清理演示文件
print()


# ===========================================================================
# 【6】权重存储格式对比表（课案表格）
# ===========================================================================
print("-" * 78)
print("【6】权重存储格式对比（课案表格）")
print("-" * 78)

format_rows = [
    ("PyTorch (.pth/.pt/.bin)", "PyTorch 官方格式", "灵活，生态兼容", "模型开发与训练", "安全风险，勿加载不可信文件"),
    ("Safetensors (.safetensors)", "Hugging Face 安全格式", "安全，加载快", "模型分享与下载", "纯权重数据，无代码"),
    ("GGUF (.gguf)", "CPU 优化格式", "CPU 友好，单文件管理", "本地 CPU 部署", "需量化模型以优化性能"),
    ("量化格式 (.q4_0.bin 等)", "模型量化减体积", "减小体积和内存占用", "边缘设备部署", "常用 GGUF 实现"),
    ("分片格式 (...00001-of-00002.bin)", "模型分片存储", "解决大文件存储问题", "超大模型管理", "基于 .bin 或 .safetensors"),
    ("ONNX (.onnx)", "跨框架交换格式", "跨平台兼容", "跨框架部署", "转换可能需调试"),
    ("TensorFlow (.pb/.ckpt/.h5)", "TensorFlow 原生格式", "功能全面，生态集成", "TensorFlow 生态开发", "与 PyTorch 不兼容，需转换"),
]
header = f"{'格式 (扩展名)':<32}{'概括':<22}{'优点':<22}{'场景':<20}关键点"
print(header)
print("-" * len(header))
for row in format_rows:
    print(f"{row[0]:<32}{row[1]:<22}{row[2]:<22}{row[3]:<20}{row[4]}")

print("\n补充说明：")
print("  · 本脚本用的就是第 1 行：torch.save(state_dict) → .pt，文件名里的 .pt/.pth/.bin 只是习惯，无格式差异。")
print("  · Safetensors 的核心优势是「格式里不允许放可执行代码」，所以加载不可信权重时更安全；")
print("    本机没有安装 safetensors，因此这里只做文字对比，不写 import（否则会因为找不到该包而失败）。")
print("  · ONNX 导出需要 torch.onnx（PyTorch 自带），但完整验证需要 onnxruntime，本机未安装，故不演示导出。")
print()


# ===========================================================================
# 【7】四种精度的位数结构
# ===========================================================================
print("-" * 78)
print("【7】精度类型的位数结构：FP32(1+8+23) / FP16(1+5+10) / BF16(1+8+7) / INT8")
print("-" * 78)

# ---- 原理：IEEE 754 浮点数的三段结构 ----
# 一个浮点数 = 符号位 s + 指数位 e + 尾数位 m，表示的值是
#     (-1)^s × 1.m × 2^(E - bias)
#     · 指数位决定**动态范围**（能表示多大/多小）——指数每多 1 位，范围翻一倍。
#     · 尾数位决定**精度**（相邻两个数之间的间隔，即机器精度 eps = 2^(-尾数位)）。
# 三者的取舍：
#     FP32：1 + 8 + 23。指数 bias=127，尾数 23 位 → eps ≈ 1.19e-7，范围 ±3.4e38。稳但占 4 字节。
#     FP16：1 + 5 + 10。指数 bias=15， 尾数 10 位 → eps ≈ 9.77e-4，范围只到 ±65504。
#           范围小是它的致命伤：训练时梯度/激活很容易超过 65504 变成 inf（要配 loss scaling）。
#     BF16：1 + 8 + 7。 指数位和 FP32 一样多（bias 相同）→ 动态范围与 FP32 完全相同；
#           但尾数只有 7 位 → eps ≈ 7.81e-3，精度比 FP16 还差。
#           深度学习对「范围」比对「精度」敏感（梯度可能有极小的值），所以 BF16 通常比 FP16 更好用，
#           而且和 FP32 互转只是截断尾数，硬件实现简单，这也是 TPU/新 GPU 主推 BF16 的原因。
#     INT8：不是浮点，是整数。1 字节表示 256 个离散值（有符号 -128~127）。
#           要表示实数必须配 scale/zero_point 做仿射量化，精度由量化粒度决定。
fp_info = [
    ("FP32", 1, 8, 23, 127, 4),
    ("FP16", 1, 5, 10, 15, 2),
    ("BF16", 1, 8, 7, 127, 2),
]
print(f"{'类型':<8}{'总位':<7}{'符号+指数+尾数':<18}{'指数bias':<11}{'eps=2^-尾数':<16}{'理论最大指数':<14}{'字节'}")
for name, s, e, m, bias, nbytes in fp_info:
    eps = 2.0 ** (-m)
    max_exp = (2 ** e - 2) - bias                      # 最大规格化指数（全 1 指数留给 inf/nan）
    print(f"{name:<8}{s + e + m:<7}{f'{s}+{e}+{m}':<18}{bias:<11}{eps:<16.3e}{f'2^{max_exp}':<14}{nbytes}")

print("\n用 PyTorch 的 finfo 核对（这就是框架自己认为的范围）：")
for name, dtype in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
    fi = torch.finfo(dtype)
    print(f"  {name:<9} eps={fi.eps:.3e}  min={fi.min:.3e}  max={fi.max:.3e}  "
          f"tiny(最小正规格化数)={fi.tiny:.3e}  bits={fi.bits}")
ii = torch.iinfo(torch.int8)
print(f"  int8      min={ii.min}  max={ii.max}  bits={ii.bits}（整数，无 eps 概念）")
print()


# ===========================================================================
# 【8】同一个张量的四种精度：内存与数值误差
# ===========================================================================
print("-" * 78)
print("【8】同一个张量转成四种精度：element_size / 占用 KB / 数值误差")
print("-" * 78)

# 构造一个「尺度差异很大」的张量，好让各种精度的误差都暴露出来：
#   · 1/3 = 0.333... 无限循环 → 任何浮点都只能近似
#   · 1e-5 很小         → FP16 的 tiny≈6.1e-5，会直接下溢成 0
#   · 巨大的值           → 单独在第 9 节演示溢出
base = torch.tensor([1.0 / 3.0, 1e-5, 0.1, 1234.5678, 1e-8, 2.0 / 7.0], dtype=torch.float32)
print(f"原始张量（float32）：")
for v in base.tolist():
    print(f"    {v!r}")
print(f"  numel = {base.numel()}，element_size = {base.element_size()} 字节")

dtypes = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("int8", torch.int8),
]
print(f"\n{'dtype':<10}{'element_size':<14}{'numel':<8}{'占用 KB':<12}{'可表示范围':<34}{'最大绝对误差'}")
print("-" * 100)
int8_scale = base.abs().max().item() / 127.0          # INT8 量化需要的 scale（见第 11 节）
for name, dtype in dtypes:
    if name == "int8":
        # 整数类型不能直接存小数，必须先量化：q = round(x / scale)，再反量化 x̂ = q * scale
        q = torch.clamp(torch.round(base / int8_scale), -128, 127).to(torch.int8)
        deq = q.float() * int8_scale
        err = (deq - base).abs().max().item()
        rng_text = f"[{torch.iinfo(torch.int8).min}, {torch.iinfo(torch.int8).max}]（整数）"
    else:
        cast = base.to(dtype)
        deq = cast.float()
        err = (deq - base).abs().max().item()
        fi = torch.finfo(dtype)
        rng_text = f"±[{fi.tiny:.1e}, {fi.max:.2e}]"
    kb = base.numel() * torch.empty(0, dtype=dtype).element_size() / 1024
    print(f"{name:<10}{torch.empty(0, dtype=dtype).element_size():<14}{base.numel():<8}{kb:<12.6f}{rng_text:<34}{err:.3e}")

print("\n逐元素看 FP16 / BF16 到底丢了多少（这才是『精度损失』的真实样子）：")
f32_t = base
f16_t = base.to(torch.float16).float()
bf16_t = base.to(torch.bfloat16).float()
print(f"{'原值(float32)':<22}{'FP16 存回来的值':<22}{'FP16 绝对误差':<20}{'BF16 存回来的值':<22}{'BF16 绝对误差'}")
for i in range(base.numel()):
    a, b, c = f32_t[i].item(), f16_t[i].item(), bf16_t[i].item()
    print(f"{a:<22.10g}{b:<22.10g}{abs(b - a):<20.3e}{c:<22.10g}{abs(c - a):.3e}")

print("\n★ 关键现象 1（下溢）：FP16 能表示的绝对值下限约 6e-8，再小就直接变成 0。")
print(f"   float32 的 1e-8 → float16 = {torch.tensor([1e-8]).to(torch.float16).item()}（完全丢失）")
print(f"   1e-10 → float16 = {torch.tensor([1e-10]).to(torch.float16).item()}，1e-30 → "
      f"{torch.tensor([1e-30]).to(torch.float16).item()}")
print(f"   补充一个容易搞错的点：1e-5 其实**不会**变成 0，它会以「非规格化数」表示——")
print(f"   float32 的 1e-5 → float16 = {torch.tensor([1e-5]).to(torch.float16).item():.6e}，")
print(f"   它小于最小正规格化数 {torch.finfo(torch.float16).tiny:.3e}，但大于非规格化下限 "
      f"2^-24 = {2 ** -24:.3e}，所以还能勉强表示，只是有效位数大幅减少。")
print("   → 这就是为什么要做 loss scaling：把梯度整体放大到 FP16 的『舒适区』再算，")
print("     否则大量小梯度会掉进非规格化区甚至变成 0，模型就学不动了。")
print("★ 关键现象 2：BF16 对 1234.5678 的误差比 FP16 大，因为它的尾数只有 7 位。")
print(f"   FP16 尾数 10 位对应的相对精度 ~ {2 ** -11:.2e}，BF16 尾数 7 位对应的相对精度 ~ {2 ** -8:.2e}（差 8 倍）")
print()


# ===========================================================================
# 【9】FP16 的坑：下溢与溢出
# ===========================================================================
print("-" * 78)
print("【9】FP16 的两个坑：小值下溢成 0，大值溢出成 inf")
print("-" * 78)

print("坑一：下溢（underflow）——小到一定程度直接变 0")
for x in (1e-3, 1e-5, 6.2e-5, 6.0e-5, 1e-7, 1e-8, 1e-12):
    x16 = torch.tensor([x]).to(torch.float16)
    flag = "← 变成 0 了！" if x16.item() == 0.0 else ""
    print(f"  float16({x:.1e}) = {x16.item():>13.6e}   {flag}")
print(f"  FP16 最小正规格化数 tiny = {torch.finfo(torch.float16).tiny:.3e}；")
print(f"  比 tiny 小的数靠「非规格化数」兜底，最小能表示 2^-24 = {2 ** -24:.3e}；再小就一律变 0。")

print("\n坑二：溢出（overflow）——超过 65504 变 inf")
# 注意：这里**故意不**真的执行 torch.tensor([70000.0], dtype=torch.float16)，
#       因为 PyTorch 会为此打一条 UserWarning（虽然结果是 inf 而不是报错）。
#       为了保持脚本「零警告、零 Traceback」，这里用算术推演 + inf 的构造来等价演示。
fi16 = torch.finfo(torch.float16)
overflow_demo = torch.tensor([float("inf")], dtype=torch.float16)     # 不经过转换，直接构造 inf
print(f"  FP16 最大值 max = {fi16.max:.1f}；如果要存 70000.0，因为它 > 65504，结果会是 {overflow_demo.item()}")
print(f"  （等价写法 torch.tensor([70000.0], dtype=torch.float16) 会得到 inf，但会附带一条警告，")
print(f"    为了保持输出干净，本脚本用 float('inf') 直接构造同样的值来演示。）")
print(f"  危险的推论：inf 参与运算会「污染」整条链：inf * 0 = {overflow_demo.item() * 0}（是 nan，不是 0！）")
print(f"               inf - inf = {overflow_demo.item() - overflow_demo.item()}，一旦出现 nan，")
print(f"               梯度会全变 nan，loss 打印出来就是 nan，这是训练崩掉最常见的现象之一。")

# 正确做法：用 float32 中转 + 检查有限性
safe = torch.tensor([70000.0], dtype=torch.float32)
print(f"\n  正确的检查方式：torch.isfinite(float32 张量) = {torch.isfinite(safe).tolist()}")
print(f"  如果真的必须降到 float16，先缩放（除以一个足够大的常数）再转、用完再乘回来：")
shrink_factor = 1e5                                        # 关键是让缩放后的值落进 FP16 的范围
scaled = (safe / shrink_factor).to(torch.float16).float() * shrink_factor
print(f"    70000 / {shrink_factor:.0e} = {(safe / shrink_factor).item():.4f}（在 FP16 范围内）")
print(f"    转 float16 再乘回来 = {scaled.item():.1f}（不再溢出；代价是精度只到 2 位有效数字）")
print(f"  工程写法：先 loss_scale 把梯度放大到安全区再转 FP16，反传前再缩回去")
print(f"            （AMP 的 loss scaling 就是干这个，GradScaler 会自动挑一个合适的倍数）。")

# 用 finfo 把「什么时候会出问题」写成可复用的判断
def check_float16_range(t: torch.Tensor) -> str:
    """判断一个张量能不能安全地放进 float16：只看范围，不看精度。"""
    fi = torch.finfo(torch.float16)
    amax = t.abs().max().item()
    if not math.isfinite(amax):
        return "含 inf/nan，无法转换"
    if amax > fi.max:
        return f"会溢出（最大值 {amax:.3e} > {fi.max:.1f}）"
    if 0 < amax < fi.tiny:
        return f"会整体下溢（最大值 {amax:.3e} < {fi.tiny:.3e}）"
    return "范围安全（但精度仍可能损失）"

for t, label in ((torch.tensor([70000.0]), "70000"), (torch.tensor([1e-6]), "1e-6"), (torch.tensor([1.5]), "1.5")):
    print(f"  check_float16_range({label:<7}) → {check_float16_range(t)}")
print()


# ===========================================================================
# 【10】BF16：范围大但精度低
# ===========================================================================
print("-" * 78)
print("【10】BF16：动态范围与 FP32 相同，但精度明显更差")
print("-" * 78)

# 用一个跨越 20 个数量级的张量说明「范围」
span = torch.tensor([1e-30, 1e-10, 1e-5, 1.0, 1e5, 1e10, 1e30], dtype=torch.float32)
print("同一个跨 60 个数量级的张量，分别转 FP16 / BF16：")
print(f"{'原值':<12}{'FP16':<18}{'BF16':<18}{'说明'}")
for v in span.tolist():
    t = torch.tensor([v], dtype=torch.float32)
    f16 = t.to(torch.float16).item()
    bf = t.to(torch.bfloat16).item()
    note = "FP16 无法表示" if (f16 in (float("inf"), 0.0)) else ("BF16 仍能表示" if bf != 0 and math.isfinite(bf) else "两者都不可")
    print(f"{v:<12.0e}{f16:<18.6e}{bf:<18.6e}{note}")
print(f"\n结论：FP16 只覆盖 [{torch.finfo(torch.float16).tiny:.2e}, {torch.finfo(torch.float16).max:.2e}]，")
print(f"      BF16 覆盖 [{torch.finfo(torch.bfloat16).tiny:.2e}, {torch.finfo(torch.bfloat16).max:.2e}]，和 FP32 完全一致。")

print("\n精度对比实验：同一个 1/3，用 FP16 和 BF16 各自存一遍，看谁更接近真值")
one_third = torch.tensor([1.0 / 3.0], dtype=torch.float32)
true_value = 1.0 / 3.0
f16_val = one_third.to(torch.float16).float().item()
bf16_val = one_third.to(torch.bfloat16).float().item()
print(f"  真值（Python float64） = {true_value:.17f}")
print(f"  FP16  存回来            = {f16_val:.17f}  误差 {abs(f16_val - true_value):.3e}")
print(f"  BF16  存回来            = {bf16_val:.17f}  误差 {abs(bf16_val - true_value):.3e}")
print(f"  → BF16 的误差是 FP16 的 {abs(bf16_val - true_value) / abs(f16_val - true_value):.1f} 倍（尾数少 3 位 → 差约 8 倍）")

# 再做一个「大量小数累加」的实验，说明低精度误差会随归约规模累积
n_accumulate = 200_000
small = torch.full((n_accumulate,), 1e-5)                    # 真值应为 200000 × 1e-5 = 2.0
sum_f32 = small.sum().item()
sum_f16 = small.to(torch.float16).sum().item()
sum_bf16 = small.to(torch.bfloat16).sum().item()
print(f"\n累加 {n_accumulate} 个 1e-5（真值应为 2.0）：")
print(f"  float32 直接累加    = {sum_f32:.8f}")
print(f"  先转 float16 再累加 = {sum_f16:.8f}   绝对误差 {abs(sum_f16 - sum_f32):.3e}")
print(f"  先转 bfloat16 再累加= {sum_bf16:.8f}   绝对误差 {abs(sum_bf16 - sum_f32):.3e}")

# 换一组「更不巧」的数据：真值同样接近 2.0，但单个数值不是 2 的幂次倍，误差不会被凑巧抵消
small2 = torch.full((n_accumulate,), 1.3e-5)
s2_f32 = small2.sum().item()
s2_f16 = small2.to(torch.float16).sum().item()
s2_bf16 = small2.to(torch.bfloat16).sum().item()
print(f"\n换一组数据：累加 {n_accumulate} 个 1.3e-5（真值应为 {n_accumulate * 1.3e-5:.6f}）：")
print(f"  float32 直接累加    = {s2_f32:.8f}")
print(f"  先转 float16 再累加 = {s2_f16:.8f}   相对误差 {abs(s2_f16 - s2_f32) / s2_f32:.3e}")
print(f"  先转 bfloat16 再累加= {s2_bf16:.8f}   相对误差 {abs(s2_bf16 - s2_f32) / s2_f32:.3e}")

print("\n  结论要说得准确一点：低精度会引入误差，但**具体哪一组更大，取决于数值本身**——")
print("  · FP16 尾数多（相对精度高），但范围窄，容易下溢/溢出；")
print("  · BF16 尾数少（单个数值的相对误差更大），但范围与 FP32 相同，不会溢出；")
print("  · 加上 CPU 的求和是「分层归约」，误差会不会被凑巧抵消有一定偶然性。")
print("     所以工程上不要靠猜：混合精度训练里把 loss 和梯度归约留在 FP32（这就是「混合」二字的含义），")
print("     必要时用 float64 做基准去实测自己的模型在低精度下究竟掉多少精度。")
print()


# ===========================================================================
# 【11】INT8 量化与反量化
# ===========================================================================
print("-" * 78)
print("【11】INT8 量化：scale / zero_point 与反量化误差")
print("-" * 78)

# ---- 原理 ----
# 仿射量化（affine quantization）：
#     量化：   q = clamp(round(x / scale) + zero_point, qmin, qmax)
#     反量化： x̂ = (q - zero_point) * scale
# 其中 scale 决定「一个整数刻度代表多大的实数」，zero_point 让实数 0 能被精确表示。
#   · 对称量化（zero_point=0）：适合权重（近似以 0 为中心分布）。
#   · 非对称量化：适合 ReLU 之后的激活（分布非负，用满 0..255）。
# 误差来源有两个：round（舍入误差 ≤ scale/2）和 clamp（超出范围的值被截断）。
w_real = torch.tensor([-1.5, -0.7, 0.0, 0.3, 0.9, 2.1, 5.0], dtype=torch.float32)
scale = w_real.abs().max().item() / 127.0                     # 对称量化的 scale
q = torch.clamp(torch.round(w_real / scale), -128, 127).to(torch.int8)
deq = q.float() * scale
print(f"权重（float32）= {w_real.tolist()}")
print(f"选定 scale = max|x| / 127 = {scale:.6f}")
print(f"量化后的 int8  = {q.tolist()}")
print(f"反量化回来     = {[round(v, 5) for v in deq.tolist()]}")
print(f"{'原值':<10}{'int8':<8}{'反量化':<12}{'绝对误差':<12}{'相对误差'}")
for i in range(w_real.numel()):
    a, qi, b = w_real[i].item(), q[i].item(), deq[i].item()
    rel = abs(b - a) / abs(a) if a != 0 else 0.0
    print(f"{a:<10.4f}{qi:<8d}{b:<12.5f}{abs(b - a):<12.5f}{rel:.4%}")
print(f"\n理论最大误差 = scale/2 = {scale / 2:.6f}，实测最大误差 = {(deq - w_real).abs().max().item():.6f}（吻合）")
print(f"压缩比：float32 4 字节 → int8 1 字节，缩小 4 倍")
print(f"  例如一个 100 万参数的模型：float32 = {1e6 * 4 / 1024 / 1024:.1f} MB，int8 = {1e6 * 1 / 1024 / 1024:.1f} MB")
print("\n有符号 vs 无符号（课案表格）：")
print(f"  int8  范围 [{torch.iinfo(torch.int8).min}, {torch.iinfo(torch.int8).max}]，适合对称的权重分布")
print(f"  uint8 范围 [{torch.iinfo(torch.uint8).min}, {torch.iinfo(torch.uint8).max}]，适合非负的激活分布")
print("  低比特 4-bit 范围 [-8, 7]、2-bit 范围 [-2, 1]：能表示的状态只剩 16 / 4 个，精度损失极大，")
print("  但配合分组量化（每 32/64 个权重共用一组 scale）在 LLM 上仍可用，这就是 GGUF 的 q4_0/q4_K 等格式。")
print()


# ===========================================================================
# 【12】混合精度 autocast：CPU 上的可用性
# ===========================================================================
print("-" * 78)
print("【12】混合精度 torch.autocast：本机 CPU 版的行为")
print("-" * 78)

# ---- 原理 ----
# 混合精度 = 计算用低精度（快、省显存）+ 关键位置保留高精度（稳）。典型配置：
#     autocast(dtype=torch.float16)：GPU 上的标准 AMP。矩阵乘法/卷积走 FP16，
#                                    但 loss、softmax、归约等仍用 FP32（这就是「混合」）。
#     autocast(dtype=torch.bfloat16)：CPU 与新版 GPU 都能用，范围大所以不需要 loss scaling。
# autocast 是**上下文管理器**：进入后 PyTorch 会自动为每个算子挑选合适精度，
# 不需要手动 .half()。注意：autocast 只管「算子用什么精度算」，
# 模型参数的 dtype 不变（所以 weight 仍是 float32，不会占更少显存——省显存要靠 GradScaler 或手动 .half()）。
x_amp = torch.randn(8, 6)
amp_ok = False
model.eval()                                          # BatchNorm 在 eval 下不做批统计，避免小 batch 报错
try:
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_amp = model(x_amp)
    print(f"启动 torch.autocast(device_type='cpu', dtype=torch.bfloat16) → 成功")
    print(f"  输出 dtype = {out_amp.dtype}，shape = {tuple(out_amp.shape)}")
    print(f"  参数 dtype 仍是 {next(model.parameters()).dtype}（autocast 不改参数 dtype）")
    with torch.no_grad():
        out_fp32 = model(x_amp)
    print(f"  与纯 float32 前向的平均绝对差 = {(out_amp.float() - out_fp32).abs().mean().item():.3e}（BF16 累计的精度损失）")
    amp_ok = True
except Exception as exc:
    print(f"启动 torch.autocast(device_type='cpu', dtype=torch.bfloat16) → 抛出 {type(exc).__name__}")
    print(f"  说明：{str(exc).splitlines()[0][:140]}")
    print("  → 已捕获。CPU 版的 autocast 支持情况取决于 PyTorch 版本与 CPU 指令集，")
    print("     旧版本 CPU 只支持 bfloat16、且部分算子没有低精度内核，会直接报错。")

if not amp_ok:
    print("\n  兜底方案：不用 autocast，手动指定 float16/bfloat16 做推理（CPU 上能跑但不一定更快）：")
    model_half = copy.deepcopy(model).to(torch.bfloat16)
    with torch.no_grad():
        out_half = model_half(x_amp.to(torch.bfloat16))
    print(f"    手动 .to(bfloat16) 前向成功：输出 dtype = {out_half.dtype}，shape = {tuple(out_half.shape)}")

# 手动 float16 推理 + 说明 CPU 上为什么不一定更快
print("\n手动 float16 推理（用于说明「低精度不一定更快」）：")
model_f16 = copy.deepcopy(model).half()
with torch.no_grad():
    out_f16 = model_f16(x_amp.half())
print(f"  模型参数 dtype = {next(model_f16.parameters()).dtype}，输出 dtype = {out_f16.dtype}")
print("  · CPU 上没有 FP16 的向量化指令优势，PyTorch 往往要来回转换，速度可能比 FP32 还慢。")
print("  · 内存倒是真的减半，所以 CPU 上做 FP16 的意义主要是「省内存」，不是「提速」。")
print("  · 想省内存又稳，CPU 上更常用 bfloat16（范围同 FP32，不需要 loss scaling）。")

# 用 fp32 累加缓解低精度误差（混合精度的核心思想）
print("\n混合精度的核心思想：用低精度算，用高精度攒。")
big = torch.randn(2000, dtype=torch.float32) * 0.01
acc_f32 = big.sum().item()
acc_f16_then_up = big.half().float().sum().item()      # 先降精度，再在 fp32 里累加
acc_f16_all = big.half().sum().item()                  # 全程 fp16 累加
print(f"  float32 累加            = {acc_f32:+.6f}")
print(f"  先转 fp16 再用 fp32 累加 = {acc_f16_then_up:+.6f}   误差 {abs(acc_f16_then_up - acc_f32):.3e}")
print(f"  全程 fp16 累加          = {acc_f16_all:+.6f}   误差 {abs(acc_f16_all - acc_f32):.3e}")
print("  → 同样的低精度输入，累加器用 fp32 时误差小一个量级。这就是 autocast 保留 fp32 归约的理由。")
print()


# ===========================================================================
# 【13】绘图：内存占用与可表示范围
# ===========================================================================
print("-" * 78)
print("【13】绘图：四种精度的内存占用 + 可表示范围")
print("-" * 78)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
fig.suptitle("精度类型：内存占用、动态范围与数值精度对比", fontsize=15, fontweight="bold")

# ---- 子图 1：内存占用（以 100 万参数为例） ----
ax = axes[0]
n_param_demo = 1_000_000
names = ["float32", "float16", "bfloat16", "int8"]
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
sizes = [torch.empty(0, dtype=dt).element_size() for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int8)]
mem_mb = [n_param_demo * s / 1024 / 1024 for s in sizes]
bars = ax.bar(names, mem_mb, color=colors, edgecolor="white")
for b, s, m in zip(bars, sizes, mem_mb):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.08, f"{s} 字节/元素\n{m:.2f} MB",
            ha="center", fontsize=9)
ax.set_ylabel("100 万参数的模型大小 (MB)", fontsize=10)
ax.set_ylim(0, max(mem_mb) * 1.32)
ax.set_title("① 内存占用对比（参数个数不变）", fontsize=11)
ax.grid(alpha=0.25, axis="y")
ax.text(0.5, 0.04, "int8 相比 float32 缩小 4 倍", transform=ax.transAxes,
        ha="center", fontsize=9, color="#c00000")

# ---- 子图 2：可表示范围（log 坐标） ----
ax = axes[1]
range_data = []
for name, dtype in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
    fi = torch.finfo(dtype)
    range_data.append((name, fi.tiny, fi.max, colors[names.index(name)]))
y_pos = np.arange(len(range_data))
for i, (name, tiny, mx, color) in enumerate(range_data):
    ax.barh(i, math.log10(mx) - math.log10(tiny), left=math.log10(tiny), height=0.5,
            color=color, edgecolor="white", alpha=0.85)
    ax.text(math.log10(tiny) - 0.6, i, f"{tiny:.1e}", ha="right", va="center", fontsize=8.5, color="#333333")
    ax.text(math.log10(mx) + 0.6, i, f"{mx:.1e}", ha="left", va="center", fontsize=8.5, color="#333333")
    ax.text((math.log10(tiny) + math.log10(mx)) / 2, i + 0.38,
            f"{name}：跨度 {math.log10(mx) - math.log10(tiny):.0f} 个数量级",
            ha="center", fontsize=9, color=color)
ax.set_yticks(y_pos, [d[0] for d in range_data], fontsize=10)
ax.set_xlabel("log10(可表示的正数范围)", fontsize=10)
ax.set_xlim(-50, 45)
ax.set_ylim(-0.7, len(range_data) - 0.15)
ax.axvline(0, color="#999999", lw=0.9, ls=":")
ax.set_title("② 动态范围对比（FP16 明显更窄）", fontsize=11)
ax.grid(alpha=0.25, axis="x")
ax.text(0.02, 0.06, "FP16 范围窄 → 需 loss scaling\nBF16 范围同 FP32 → 无需缩放",
        transform=ax.transAxes, fontsize=8.5, color="#404040")

# ---- 子图 3：同一个值在不同精度下的误差 ----
ax = axes[2]
test_vals = [1.0 / 3.0, 0.1, 1e-5, 1234.5678, 2.0 / 7.0]
labels = ["1/3", "0.1", "1e-5", "1234.5678", "2/7"]
xpos = np.arange(len(test_vals))
w = 0.35
err_f16 = []
err_bf16 = []
for v in test_vals:
    t = torch.tensor([v], dtype=torch.float32)
    err_f16.append(abs(t.to(torch.float16).float().item() - v))
    err_bf16.append(abs(t.to(torch.bfloat16).float().item() - v))
err_f16 = np.maximum(np.array(err_f16), 1e-12)          # 0 取不到 log 坐标，抬到 1e-12 只用于显示
err_bf16 = np.maximum(np.array(err_bf16), 1e-12)
ax.bar(xpos - w / 2, err_f16, w, color="#dd8452", edgecolor="white", label="float16")
ax.bar(xpos + w / 2, err_bf16, w, color="#55a868", edgecolor="white", label="bfloat16")
ax.set_yscale("log")
ax.set_xticks(xpos, labels, fontsize=9)
ax.set_ylabel("绝对误差（log 坐标）", fontsize=10)
ax.set_title("③ 同一个值的表示误差", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
ax.text(0.03, 0.90, "FP16 尾数多→单值更准\nBF16 尾数少→误差约 8 倍\n（1e-5 已属非规格化区）", transform=ax.transAxes,
        fontsize=8.5, color="#c00000")

plt.tight_layout(rect=[0, 0, 1, 0.92])
save_path = OUTPUT_DIR / "04_精度类型_内存与范围对比.png"
plt.savefig(save_path, dpi=110, bbox_inches="tight")
plt.close(fig)

print(f"图已保存：{save_path}")
print(f"文件存在：{save_path.exists()}，大小：{save_path.stat().st_size / 1024:.1f} KB")
print()

# ===========================================================================
# 小结
# ===========================================================================
print("=" * 78)
print("小结")
print("=" * 78)
print("· state_dict 是有序字典，只含 parameters + buffers，不含网络结构；值和模型共享内存")
print("· load_state_dict 是原地拷贝；strict=True 全有或全无，strict=False 返回 missing/unexpected_keys")
print("· 形状不匹配 strict 也救不了；加载到新结构时 strict=False 是迁移学习的标准写法")
print("· 保存一律用 torch.save(model.state_dict())；torch.save(model) 依赖类定义且有 pickle 安全风险")
print("· 完整 checkpoint 必须带上 optimizer / scheduler / epoch / best_metric / config，才能断点续训")
print("· 格式：.pt/.pth/.bin（PyTorch）、Safetensors（安全）、GGUF（CPU 量化）、ONNX（跨框架）、TF（需转换）")
print("· 位数结构：FP32=1+8+23、FP16=1+5+10、BF16=1+8+7、INT8=8 位整数（需 scale/zero_point）")
print("· FP16 范围 ±[6.1e-5, 6.55e4]：1e-8 及更小下溢成 0、70000 溢出成 inf，所以需要 loss scaling")
print("· BF16 范围与 FP32 相同但尾数只有 7 位，精度比 FP16 差约 8 倍，好在范围够大不用缩放")
print("· 混合精度：低精度算、FP32 累加（归约/loss 保留高精度），autocast 在 CPU 上支持有限需 try/except")
print("=" * 78)
print("脚本执行完毕，退出码 0。")
