"""
对应课案章节：网络架构 → Embedding

本节知识点：
    1. 为什么要 Embedding：把离散 token（如"猫"→编号 42）映射为稠密向量，
       让神经网络能处理文字这类离散符号。
    2. 为什么不用 One-Hot：词表 50000 时是 50000 维稀疏向量（维度爆炸，内存/计算浪费），
       且任意两个不同词的 One-Hot 向量互相正交（内积恒为 0），无法表达
       "猫"和"狗"都是宠物这种语义相似性。
    3. Embedding 的本质是「查表」：权重矩阵 W 形状 (vocab_size, d_model)，
       第 i 行就是第 i 个词的向量，输入 token id 就是行号索引，前向传播 = 取对应行。
       —— 本脚本用 F.one_hot(ids, vocab_size).float() @ W 手写验证这一点（核心洞见）。
    4. 输入形状进阶：(batch, seq_len) -> (batch, seq_len, d_model)。
    5. padding_idx：被指定为 padding 的位置，其向量恒为 0 且不参与梯度更新；
       常用于变长序列对齐（短句补齐的 <PAD> 不该贡献任何语义/梯度）。
    6. max_norm（按范数裁剪词向量，稳定训练）与 nn.EmbeddingBag（把查表 + 求和/求均值
       融成一步，适合"一袋词"式的定长向量表示，省掉中间的 (batch, seq_len, d_model)）。
    7. 词向量相似度：用一个人造共现任务（skip-gram 式"由中心词预测上下文词"）
       在小词表上训练几十到上百步，语义相近的词向量就会靠近；
       用余弦相似度矩阵量化这件事。
    8. 可视化：余弦相似度热力图 + PCA（奇异值分解 SVD）降到 2 维的散点图。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\02_网络架构\\01_Embedding词嵌入.py'
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
# 后续正式依赖（注意：必须写在 matplotlib 初始化之后）
# ---------------------------------------------------------------------------
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# 随机种子统一：保证每次运行结果一致、可复现
torch.manual_seed(42)
np.random.seed(42)

# 计时器：交付要求单脚本 < 40 秒（优先 < 20 秒），最后打印总耗时
_T0 = time.perf_counter()


def section(title: str) -> None:
    """统一的章节打印，让控制台输出层次分明。"""
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# 一、One-Hot 的两个致命问题
# ===========================================================================
section("一、为什么不用 One-Hot：维度爆炸 + 正交无语义")

VOCAB_SIZE_BIG = 50000  # 课案里的词表规模
D_MODEL = 512           # 课案里的嵌入维度

# 直观算一笔账：One-Hot 需要 vocab_size 维，而且绝大多数位置是 0
one_hot_bytes = VOCAB_SIZE_BIG * 4          # float32 每个数 4 字节
embed_bytes = D_MODEL * 4                   # Embedding 每个词只要 d_model 个数
print(f"假设词表 {VOCAB_SIZE_BIG} 个词：")
print(f"  One-Hot 向量维度 = {VOCAB_SIZE_BIG}，单个词占内存 {one_hot_bytes / 1024:.1f} KB  (float32)")
print(f"  Embedding 向量维度 = {D_MODEL}，单个词占内存 {embed_bytes / 1024:.2f} KB")
print(f"  压缩倍数 ≈ {one_hot_bytes / embed_bytes:.1f} 倍，而且稠密向量每个维度都携带信息")

# 第二个问题：不同词的 One-Hot 向量严格正交 —— 无法表达语义相似
cat_oh = F.one_hot(torch.tensor(42), num_classes=VOCAB_SIZE_BIG).float()    # "猫" -> id 42
dog_oh = F.one_hot(torch.tensor(7), num_classes=VOCAB_SIZE_BIG).float()     # "狗" -> id 7
dot = torch.dot(cat_oh, dog_oh).item()                                     # 点积 = 0 表示正交
print(f"\nOne-Hot('猫') 与 One-Hot('狗') 的点积 = {dot:.1f}")
print("点积为 0 ⇒ 两个向量正交 ⇒ 余弦相似度无从谈起，"
      "模型永远学不到『猫和狗都是宠物』这种共性。")
print("Embedding 把每个词压成低维稠密向量（如 512 维），相似的词向量也相似。")

# ===========================================================================
# 二、课案原例：nn.Embedding(10000, 512) 查表得到 (4, 512)
# ===========================================================================
section("二、课案原例：nn.Embedding(10000, 512) 的前向传播")

vocab_size = 10000   # 词表大小
d_model = 512        # 每个词用 512 维向量表示

embedding = nn.Embedding(vocab_size, d_model)

# 输入：token id（整数张量，dtype 必须是 torch.long / int64）
token_ids = torch.tensor([42, 7, 9999, 0])   # 4 个 token
vec = embedding(token_ids)                   # (4, 512)，查出 4 个 512 维向量

print(f"输入 token_ids      = {token_ids.tolist()}  shape={tuple(token_ids.shape)}")
print(f"输出 vec.shape      = {tuple(vec.shape)}   # torch.Size([4, 512])")
print(f"embedding.weight.shape = {tuple(embedding.weight.shape)}  # (vocab_size, d_model)")
print(f"vec[0] 就是 weight 的第 42 行，前 8 个分量 = {vec[0, :8].tolist()}")

# ===========================================================================
# 三、核心洞见：Embedding 就是「查表」，用 One-Hot 矩阵乘法手写验证
# ===========================================================================
section("三、核心洞见：Embedding ≡ One-Hot 矩阵乘法（手写验证）")

# 数学本质推导：
#   One-Hot 向量 o_i 是长度为 vocab_size 的 one-hot 行向量（第 i 位为 1）。
#   用它去乘权重矩阵 W（形状 vocab_size × d_model）：
#       o_i @ W = W 的第 i 行
#   矩阵形式：O @ W，O 是 (N, vocab_size) 的 One-Hot 矩阵，
#   结果形状 (N, d_model)，等价于把每个 id 对应的行"挑"出来。
#   而 nn.Embedding 的 forward 恰恰就是这个"挑行"操作（底层用 index_select）。
#   所以 Embedding = 查表 = 稀疏矩阵乘法的高效实现。
W = embedding.weight.detach()                       # 取出权重矩阵 (10000, 512)
one_hot = F.one_hot(token_ids, num_classes=vocab_size).float()   # (4, 10000) 的 One-Hot 矩阵
lookup_by_matmul = one_hot @ W                      # (4, 10000) @ (10000, 512) -> (4, 512)

print(f"One-Hot 矩阵 shape = {tuple(one_hot.shape)}   # 4 个 token，每个 10000 维稀疏向量")
print(f"One-Hot 中 1 的个数 = {int(one_hot.sum().item())}   # 每行恰好 1 个 1，其余全 0")
print(f"one_hot @ W 的结果 shape = {tuple(lookup_by_matmul.shape)}")

max_err_lookup = (lookup_by_matmul - vec.detach()).abs().max().item()
print(f"One-Hot 矩阵乘法 vs nn.Embedding 的最大误差 = {max_err_lookup:.3e}"
      f"  (< 1e-6 ⇒ 两者完全等价)")
assert max_err_lookup < 1e-6, "Embedding 应当严格等价于 One-Hot 矩阵乘法"

# 换个说法：直接按行号索引，结果一模一样（这才是真正的"查表"语义）
vec_by_index = W[token_ids]                          # 高级索引：取第 42/7/9999/0 行
print(f"直接索引 W[token_ids] 与 nn.Embedding 的最大误差 = "
      f"{(vec_by_index - vec.detach()).abs().max().item():.3e}")
print("结论：前向传播没有做任何乘法，只是『按行号取行』。"
      "One-Hot 视角只是帮助我们理解『它等价于一次巨型稀疏矩阵乘法』。")

# 反过来看计算量：真正的矩阵乘法做了 4 * 10000 * 512 ≈ 2000 万次乘加，
# 而只需取 4 行共 2048 个数。这就是工程上必须用查表实现的原因。
print(f"\n用 One-Hot 做乘法需要 {token_ids.numel() * vocab_size * d_model:,} 次乘加；"
      f"查表只需要搬运 {token_ids.numel() * d_model:,} 个数。")

# 顺带看看随机初始化的词向量长什么样。
# 源码事实：nn.Embedding.reset_parameters() 里只有一行 init.normal_(self.weight)，
# 即默认用标准正态分布 N(0, 1) 初始化（不是均匀分布！）。
print(f"随机初始化词向量的均值={W.mean().item():+.4f}，标准差={W.std().item():.4f}"
      f"  # 默认 init.normal_ 即 N(0,1)，理论均值 0、标准差 1")
print("实际工程里通常改成更小的初始化（如 N(0, 0.02)），"
      "否则和深层网络的尺度不匹配，容易造成训练初期震荡。")

# ===========================================================================
# 四、输入形状进阶：(batch, seq_len) -> (batch, seq_len, d_model)
# ===========================================================================
section("四、输入形状进阶：批量 + 序列长度")

batch_size, seq_len = 2, 6                           # 2 条样本，每条 6 个 token
ids_2d = torch.randint(0, vocab_size, (batch_size, seq_len))   # (2, 6) 随机 token id
out_3d = embedding(ids_2d)                           # (2, 6, 512)
print(f"输入 ids_2d.shape  = {tuple(ids_2d.shape)}          # (batch, seq_len)")
print(f"输出 out_3d.shape  = {tuple(out_3d.shape)}   # (batch, seq_len, d_model)")
print("规律：Embedding 只把每个整数换成 d_model 维向量，"
      "在最后追加一个维度，前面的 batch/seq_len 原样保留。")

# 验证某个位置：out_3d[1, 3] 应当等于 weight 的第 ids_2d[1,3] 行
i, j = 1, 3
same_row = torch.allclose(out_3d[i, j], W[ids_2d[i, j].item()], atol=1e-6)
print(f"out_3d[{i},{j}] 是否等于 weight[{ids_2d[i, j].item()}] 行：{same_row}")

# 位置编码（Positional Encoding）的动机铺垫：Embedding 本身「与位置无关」，
# 同一个 id 出现在任何位置，查出的向量完全相同。
id_a = torch.tensor([[42, 8, 13]])
id_b = torch.tensor([[8, 42, 13]])
emb_a, emb_b = embedding(id_a), embedding(id_b)
print("同一个 token 42 出现在不同位置时向量完全相同 ⇒ Embedding 不含位置信息，"
      "序列模型需要额外叠加位置编码。")
print(f"验证：emb_a[0,0] 与 emb_b[0,1] 是否相等 = "
      f"{torch.allclose(emb_a[0, 0], emb_b[0, 1], atol=1e-6)}")

# ===========================================================================
# 五、padding_idx：补齐位恒为 0 且不参与梯度
# ===========================================================================
section("五、padding_idx：让 <PAD> 不产生语义也不产生梯度")

# 变长序列要凑成矩形张量，就得用 0 之类的 id 补齐（padding）。
# 若不处理，这些"填充词"会被当成真实词汇参与训练：
#   - 它们的向量会被更新，可能污染语义空间；
#   - 它们会进入后续层的加权求和，影响结果。
# nn.Embedding(padding_idx=0) 做两件事：
#   - 前向传播时，凡是 id == padding_idx 的位置，输出向量恒为全 0；
#   - 反向传播时，跳过这些位置，对应权重行的梯度恒为 0（不会被更新）。
pad_id = 0
emb_pad = nn.Embedding(50, 8, padding_idx=pad_id)       # 设置 padding_idx=0
emb_nopad = nn.Embedding(50, 8)                          # 不设置 padding_idx
with torch.no_grad():                                    # 手工把两张表设成完全一样，排除随机初始化干扰
    emb_nopad.weight.copy_(emb_pad.weight)

ids_with_pad = torch.tensor([3, 5, pad_id, pad_id, 9])   # 中间两个是补齐位
out_pad = emb_pad(ids_with_pad)
out_nopad = emb_nopad(ids_with_pad)

print(f"输入 ids = {ids_with_pad.tolist()}（第 2、3 个是 padding）")
print(f"padding_idx=0 时 weight[0] 全为 0：{bool(torch.all(emb_pad.weight[pad_id] == 0))}")
print(f"输出中 padding 位置是否全 0：{bool(torch.all(out_pad[2:4] == 0))}")
print(f"未设置 padding_idx 时同一位置是否全 0：{bool(torch.all(out_nopad[2:4] == 0))}"
      f"   # False：补齐位被当成了普通词")

# 再看梯度：把 padding 位和普通位的输出都求和后反传，观察各自梯度
loss_pad = out_pad.sum()
loss_pad.backward()
grad_pad_row0 = emb_pad.weight.grad[pad_id]
grad_pad_row3 = emb_pad.weight.grad[3]
print(f"\npadding_idx=0 时 weight.grad[0] 是否全 0：{bool(torch.all(grad_pad_row0 == 0))}"
      f"   # 补齐位不参与学习")
print(f"          普通词 weight.grad[3] 是否全 0：{bool(torch.all(grad_pad_row3 == 0))}"
      f"   # 普通词正常累积梯度")

emb_nopad.weight.grad = None
out_nopad.sum().backward()
print(f"未设置 padding_idx 时 weight.grad[0] 是否全 0："
      f"{bool(torch.all(emb_nopad.weight.grad[pad_id] == 0))}   # False：padding 位也在被训练")

print("\n补充：padding_idx 只影响『前向输出恒 0 + 反向梯度为 0』，"
      "它不会阻止你传入该 id；这是一种约定俗成的工程写法。")

# ===========================================================================
# 六、max_norm 与 nn.EmbeddingBag（可选进阶）
# ===========================================================================
section("六、可选进阶：max_norm 与 nn.EmbeddingBag")

# max_norm：每次前向传播前，若某个词向量的 L2 范数超过 max_norm，就把它原地缩放回 max_norm。
# 直觉：词向量范数越大，进入后续点积/softmax 时越容易造成极端 logits；
#       限制范数上界能让训练更稳定（对出现频率极高的词尤其有效）。
emb_clip = nn.Embedding(100, 16, max_norm=1.0)
with torch.no_grad():                                    # 故意造一个范数超标的词向量
    emb_clip.weight[7] = torch.randn(16) * 10
print(f"裁剪前 weight[7] 的 L2 范数 = {emb_clip.weight[7].norm().item():.4f}")
_ = emb_clip(torch.tensor([7]))                          # 前向传播会触发原地裁剪
print(f"裁剪后 weight[7] 的 L2 范数 = {emb_clip.weight[7].norm().item():.4f}"
      f"   # 被压到 max_norm=1.0")

# nn.EmbeddingBag：直接对"一袋词"的向量做聚合（求和 / 求均值），
# 不返回 (batch, seq_len, d_model) 这种三维中间结果，更省内存。
# 输入有两种形式：
#   (1) 2D 定长形式 inputs: (batch, seq_len) + mode="mean"  —— 每条样本长度相同；
#   (2) 1D 变长形式 input: (N,) + offsets: (batch+1,)      —— 每条样本长度不同。
bag = nn.EmbeddingBag(100, 16, mode="mean")
bag_ids_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
bag_out = bag(bag_ids_2d)                                # (2, 16)，每行是 3 个词向量的平均
print(f"\nEmbeddingBag 输入 shape = {tuple(bag_ids_2d.shape)} -> 输出 shape = {tuple(bag_out.shape)}"
      f"   # 序列维被聚合掉")
manual_mean = bag.weight[[1, 2, 3]].mean(dim=0)          # 手工求均值验证
print(f"与手工 weight[[1,2,3]].mean(0) 的最大误差 = "
      f"{(bag_out[0] - manual_mean).abs().max().item():.3e}")
print("EmbeddingBag 常用于词袋分类：文本分类只关心整句的定长表示，"
      "不需要保留每个位置。")

del embedding, emb_pad, emb_nopad, emb_clip, bag, W, one_hot, lookup_by_matmul
del vec, out_3d, out_pad, out_nopad, emb_a, emb_b      # 及时释放，避免和小词表实验抢内存

# ===========================================================================
# 七、词向量相似度：在人造共现任务上把语义相近的词"拉近"
# ===========================================================================
section("七、词向量相似度：小词表 + 人造共现任务训练")

# 设计一个只有 8 个词的"玩具词表"，并人为规定两个语义类别：
#   类别 0（动物）：猫 狗 鱼 鸟     类别 1（交通工具）：汽车 火车 飞机 轮船
# 训练任务（skip-gram 式）：给定中心词，预测它周围出现的上下文词。
# 我们人造的"语料"规则是：上下文词与中心词大概率同类（80%），
# 只有 20% 概率跨类。模型为了降低交叉熵损失，只能把同类词的向量拉近
# （因为输出层是 softmax(dot(中心词向量, 上下文词向量))，点积大 ⇒ 概率高）。
# 于是训练结束后：同类词的余弦相似度显著高于跨类词。
word_list = ["猫", "狗", "鱼", "鸟", "汽车", "火车", "飞机", "轮船"]
small_vocab = len(word_list)                 # 8
SMALL_DIM = 16                               # 词向量维度（小到可以秒训）
category = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])   # 每个词所属语义类别

torch.manual_seed(42)                        # 复现固定
# 构造训练对：先抽中心词，再按 80% 概率从同类里抽上下文词、20% 概率从异类里抽
n_pairs = 2048
center = torch.randint(0, small_vocab, (n_pairs,))            # 中心词 id
same_class = torch.rand(n_pairs) < 0.85                       # 是否采同类上下文
# 同类：在 4 个同类词里选「除自己以外」的 3 个之一 -> 不允许自己预测自己
offset = torch.randint(1, 4, (n_pairs,))                      # 1/2/3
context_same = (center + offset) % 4 + torch.where(category[center] == 0,
                                                   torch.zeros_like(center),
                                                   torch.full_like(center, 4))
# 跨类：换成另一个类别里随机的词
context_other = torch.where(category[center] == 0, center % 4 + 4, center % 4)
context = torch.where(same_class, context_same, context_other)
print(f"词表：{word_list}")
print(f"类别：动物 {word_list[:4]} / 交通工具 {word_list[4:]}")
print(f"训练对数量 = {n_pairs}，其中同类占 {same_class.float().mean().item():.1%}"
      f"（人造共现规则：同类概率 85%）")

class SkipGramToy(nn.Module):
    """玩具版 skip-gram：Embedding 查表 + 与同一张词向量表做点积 -> softmax 预测上下文词。

    关键设计：**权重共享（weight tying）**，输出层的权重就是输入 Embedding 的权重矩阵。
    原始 word2vec 与很多现代模型（如 Transformer 的 lm_head）都这么做：
        logits = h @ Eᵀ        （h 是中心词向量，E 是整张词向量表）
    这样"预测上下文"就变成"让中心词向量与上下文词向量的点积尽可能大"，
    梯度会直接把语义相近的词拉近 —— 学到的词向量因此具有可解释的相似度结构。
    """

    def __init__(self, vocab: int, dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)      # (vocab, dim) 唯一的词向量表

    def forward(self, center_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(center_ids)                 # (B,) -> (B, dim)  查表
        logits = h @ self.embed.weight.t()         # (B, dim) @ (dim, vocab) -> (B, vocab)
        return logits


torch.manual_seed(42)
toy_model = SkipGramToy(small_vocab, SMALL_DIM)
# 学习率刻意取小（0.01）：玩具任务里若学习率太大（如 0.5），
# 8 个词的向量会被极度压缩到"两个方向"上，同类词相似度饱和到 1.000，
# 失去观察意义；小学习率能让同类词在球面上保留各自的位置，形成自然的簇。
toy_opt = torch.optim.Adam(toy_model.parameters(), lr=0.01)
toy_loss_fn = nn.CrossEntropyLoss()

N_STEP, BATCH = 100, 256                       # 100 步 × 256 → 秒级完成
toy_model.train()
first_loss = last_loss = None
for step in range(N_STEP):
    idx = torch.randint(0, n_pairs, (BATCH,))          # 随机抽一批训练对
    x_batch, y_batch = center[idx], context[idx]
    loss = toy_loss_fn(toy_model(x_batch), y_batch)
    toy_opt.zero_grad()
    loss.backward()
    toy_opt.step()
    if step == 0:
        first_loss = loss.item()
    last_loss = loss.item()
print(f"\n训练 {N_STEP} 步：首步 loss = {first_loss:.4f} → 末步 loss = {last_loss:.4f}"
      f"（随机瞎猜 8 类的理论损失 = ln(8) = {np.log(8):.4f}）")

# 取训练好的词向量矩阵，逐行做 L2 归一化，再算余弦相似度矩阵：S = Ŵ @ Ŵᵀ
toy_model.eval()
with torch.no_grad():
    emb_matrix = toy_model.embed.weight.detach().clone()          # (8, 16)
    emb_norm = emb_matrix / emb_matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos_sim = emb_norm @ emb_norm.t()                             # (8, 8) 余弦相似度
cos_np = cos_sim.numpy()

print("\n余弦相似度矩阵（保留 2 位小数，对角线恒为 1）：")
header = "        " + "".join(f"{w:>7}" for w in word_list)
print(header)
for i, w in enumerate(word_list):
    print(f"{w:>6}  " + "".join(f"{cos_np[i, j]:>7.2f}" for j in range(small_vocab)))

# 找出相似度最高的若干个词对（排除自己和自己）
pairs = []
for i in range(small_vocab):
    for j in range(i + 1, small_vocab):
        pairs.append((cos_np[i, j], i, j))
pairs.sort(reverse=True)

print("\n相似度最高的 6 个词对：")
for s, i, j in pairs[:6]:
    tag = "同类" if category[i].item() == category[j].item() else "跨类"
    print(f"  {word_list[i]} - {word_list[j]}：{s:+.4f}  [{tag}]")
print("相似度最低的 3 个词对：")
for s, i, j in pairs[-3:]:
    tag = "同类" if category[i].item() == category[j].item() else "跨类"
    print(f"  {word_list[i]} - {word_list[j]}：{s:+.4f}  [{tag}]")

same_sims = np.array([s for s, i, j in pairs if category[i].item() == category[j].item()])
diff_sims = np.array([s for s, i, j in pairs if category[i].item() != category[j].item()])
print(f"\n同类词对平均余弦相似度 = {same_sims.mean():+.4f}，"
      f"跨类词对平均余弦相似度 = {diff_sims.mean():+.4f}")
print(f"差距 = {same_sims.mean() - diff_sims.mean():+.4f} > 0 "
      f"⇒『语义相近的词，学出来的向量也相近』得到验证。")

# 更直观的指标：每个词「最相似的另一个词」是否属于同一语义类别
hit = 0
print("\n每个词最相似的邻居：")
for i, w in enumerate(word_list):
    others = [(cos_np[i, j], j) for j in range(small_vocab) if j != i]
    best_s, best_j = max(others)
    ok = category[i].item() == category[best_j].item()
    hit += int(ok)
    print(f"  {w:<4} -> {word_list[best_j]:<4}（相似度 {best_s:+.3f}，"
          f"{'同类 ✓' if ok else '跨类 ✗'}）")
print(f"最近邻同类命中率 = {hit}/{small_vocab}，"
      f"随机猜的期望只有 {3 / 7:.1%}（同类可选 3 个 / 其余共 7 个）")

# ===========================================================================
# 八、可视化：余弦相似度热力图 + PCA(SVD) 二维散点
# ===========================================================================
section("八、可视化：余弦相似度热力图 + PCA 降维散点图")

# PCA 的两种等价说法：
#   1) 对中心化后的数据做奇异值分解 X = U S Vᵀ，取前 k 个主方向 V[:, :k]，
#      投影 Z = X @ V[:, :k] 就是前 k 个主成分。
#   2) 等价于求协方差矩阵 XᵀX 的特征向量。
# 这里手写 SVD 版 PCA（不引入 umap；也可以用 sklearn.decomposition.PCA，结果一致）。
emb_centered = emb_matrix.numpy() - emb_matrix.numpy().mean(axis=0, keepdims=True)  # 中心化
U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)    # 薄 SVD
Z = emb_centered @ Vt[:2].T                                    # (8, 2) 降到二维
explained = (S[:2] ** 2) / (S ** 2).sum()                      # 前两个主成分解释的方差比例
print(f"SVD 奇异值 = {np.round(S, 3).tolist()}")
print(f"前 2 个主成分解释的方差比例 = {explained.sum():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：余弦相似度热力图
im = axes[0].imshow(cos_np, cmap="coolwarm", vmin=-1, vmax=1)
axes[0].set_xticks(range(small_vocab))
axes[0].set_yticks(range(small_vocab))
axes[0].set_xticklabels(word_list)
axes[0].set_yticklabels(word_list)
for i in range(small_vocab):
    for j in range(small_vocab):
        axes[0].text(j, i, f"{cos_np[i, j]:.2f}", ha="center", va="center",
                     fontsize=8, color="black")
axes[0].set_title("8 个词的余弦相似度矩阵\n（左上 4×4 为动物，右下 4×4 为交通工具）")
axes[0].set_xlabel("词")
axes[0].set_ylabel("词")
fig.colorbar(im, ax=axes[0], shrink=0.85, label="cosine similarity")

# 右图：PCA 二维散点，按真实语义类别上色
colors = ["#d62728", "#1f77b4"]
for c in (0, 1):
    sel = category.numpy() == c
    cname = "动物" if c == 0 else "交通工具"
    axes[1].scatter(Z[sel, 0], Z[sel, 1], s=180, c=colors[c], label=cname,
                    edgecolors="white", linewidths=1.5, zorder=3)
for i, w in enumerate(word_list):
    axes[1].annotate(w, (Z[i, 0], Z[i, 1]), textcoords="offset points",
                     xytext=(8, 6), fontsize=11)
axes[1].axhline(0, color="gray", lw=0.6, ls="--")
axes[1].axvline(0, color="gray", lw=0.6, ls="--")
axes[1].set_title(f"词向量的 PCA 二维投影（SVD 手写实现，解释方差 {explained.sum():.0%}）")
axes[1].set_xlabel("主成分 1")
axes[1].set_ylabel("主成分 2")
axes[1].legend()
axes[1].grid(alpha=0.25)

fig.suptitle("Embedding 词向量：相似的词，向量也相似", fontsize=14)
fig.tight_layout()
save_path = OUTPUT_DIR / "02网络架构_01_Embedding词向量相似度.png"
fig.savefig(save_path, dpi=120)
plt.close(fig)
print(f"\n图片已保存：{save_path}")

# ===========================================================================
# 九、本节小结
# ===========================================================================
section("九、本节小结")
print("1. Embedding 把离散 token id 映射为 d_model 维稠密向量，解决了 One-Hot 的")
print("   维度爆炸（50000 维稀疏）与正交无语义（猫/狗点积为 0）两大问题。")
print("2. 本质是查表：W 形状 (vocab_size, d_model)，第 i 行就是第 i 个词的向量；")
print("   已用 F.one_hot(ids, V).float() @ W 与 nn.Embedding 的 0 误差验证。")
print("3. 形状规律：(batch, seq_len) -> (batch, seq_len, d_model)。")
print("4. padding_idx 让补齐位向量恒 0 且梯度恒 0；max_norm 限制范数上界；")
print("   EmbeddingBag 直接输出聚合成的一袋词定长表示。")
print("5. 在 8 词人造共现任务上训练 100 步后，同类词余弦相似度明显高于跨类词，")
print("   最近邻也几乎总是同类词，说明『相似的词向量也相似』是任务驱动的")
print("   梯度下降学出来的，而不是人工写死的。")

print(f"\n脚本总耗时：{time.perf_counter() - _T0:.2f} 秒")
print("脚本正常结束。")
