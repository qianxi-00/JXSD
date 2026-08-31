# ========== 环境配置 ==========
import pandas as pd
import numpy as np
from collections import Counter  # 用于统计元素出现频次
import matplotlib.pyplot as plt

# 样式设置必须在字体设置之前
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ========== 数据准备 ==========
"""
业务场景：电商平台用户行为路径分析

分析目的：
1. 还原用户从首页到下单的真实操作轨迹
2. 识别高频路径和关键转化节点
3. 发现路径中的流失点和优化机会

数据模拟逻辑：
- 模拟500个用户会话(sessions)
- 每个会话代表一次完整的用户访问轨迹
- 用户分为三种类型：搜索型、浏览型、目标明确型
"""

# np.random.seed(42): 设置随机数种子
# 参数42: 种子值，相同种子产生相同的随机数序列
# 作用: 确保结果可复现
# 返回值: 无
np.random.seed(42)

n_sessions = 500    # 模拟500个用户会话

# 定义路径中的所有可能节点
# 节点按用户操作流程大致排序：首页 -> 搜索/分类导航 -> 商品详情 -> 加购/收藏 -> 下单 -> 支付
nodes = ['首页', '搜索', '分类导航', '商品详情', '加入购物车', '收藏', '查看评价', '领券', '提交订单', '支付']

def generate_path():
    """
    生成单个用户的行为路径

    业务逻辑：
    1. 所有用户都从首页开始
    2. 根据用户类型选择不同的浏览路径
    3. 在商品详情页后可能有附加行为（收藏、评价、领券）
    4. 部分用户会加入购物车并最终下单支付

    Returns:
        str: 用箭头连接的路径字符串，如 '首页 → 搜索 → 商品详情'
    """
    path = ['首页']  # 所有用户从首页开始

    # 随机选择用户类型，不同类型有不同的行为模式
    # 搜索型用户：通过搜索直达商品，效率最高
    # 浏览型用户：通过分类导航浏览，发现式购物
    # 目标明确型用户：直接进入商品详情，可能是复购或深度用户

    # np.random.choice(): 从列表中随机选择一个元素
    # 参数: 列表或数组
    # 返回值: 随机选择的元素
    user_type = np.random.choice(['搜索型', '浏览型', '目标明确型'])

    if user_type == '搜索型':
        path += ['搜索', '商品详情']      # 搜索型：首页→搜索→商品详情
    elif user_type == '浏览型':
        path += ['分类导航', '商品详情']  # 浏览型：首页→分类导航→商品详情
    else:
        path += ['商品详情']              # 目标明确型：首页→商品详情（直达）

    # np.random.random(): 生成[0, 1)之间的随机浮点数
    # 返回值: 0到1之间的随机小数
    # 70%概率在商品详情后执行附加行为（收藏、查看评价、领券）
    if np.random.random() > 0.3:
        # path.append(): 列表方法，在列表末尾添加元素
        # 参数: 要添加的元素
        # 返回值: 无（修改原列表）
        path.append(np.random.choice(['收藏', '查看评价', '领券']))

    # 40%概率加入购物车
    if np.random.random() > 0.6:
        path.append('加入购物车')

    # 30%概率从购物车提交订单（前提是已经加入购物车）
    if np.random.random() > 0.7 and '加入购物车' in path:
        path.append('提交订单')
        # 85%概率完成支付
        if np.random.random() > 0.15:
            path.append('支付')

    # str.join(): 字符串方法，用指定字符连接列表元素
    # ' → '.join(['首页', '搜索', '商品详情']) → '首页 → 搜索 → 商品详情'
    # 参数: 可迭代对象（列表、元组等）
    # 返回值: 连接后的字符串
    return ' → '.join(path)

# ========== 分析步骤 ==========
"""
分析方法说明：
1. 路径频次统计：统计每条完整路径出现的次数，找出高频路径
2. 节点频次统计：统计每个节点在所有路径中出现的总次数
3. 转换频次统计：统计相邻节点之间的转换次数，识别关键路径节点
"""

# 列表推导式：[expression for item in iterable]
# 生成所有用户的路径数据
# range(n_sessions): 生成0到n_sessions-1的整数序列
# _: 临时变量，表示不使用循环变量
# generate_path(): 调用函数生成一条路径
paths = [generate_path() for _ in range(n_sessions)]

# Counter(): collections.Counter，用于统计元素出现频次
# 参数: 可迭代对象
# 返回值: 字典子类，键=元素，值=频次
# 作用: 快速统计每个元素出现的次数
path_counts = Counter(paths)

# 输出Top 10高频路径
print("Top 10 用户行为路径:")

# Counter.most_common(n): 返回出现频次最高的n个元素
# 参数n: 返回的元素数量
# 返回值: 列表，元素为(元素, 频次)的元组
# enumerate(iterable, start): 为可迭代对象添加索引
# 参数start: 起始索引值（默认0）
# 返回值: 迭代器，每次返回(index, element)
for i, (path, cnt) in enumerate(path_counts.most_common(10), 1):
    print(f"{i}. [{cnt}次] {path}")

# 统计每个节点出现的总频次
all_nodes = []

# for循环遍历所有路径
for path in paths:
    # str.split(): 字符串方法，按指定分隔符拆分字符串
    # 参数: 分隔符字符串
    # 返回值: 列表，包含拆分后的子字符串
    # '首页 → 搜索 → 商品详情'.split(' → ') → ['首页', '搜索', '商品详情']
    all_nodes.extend(path.split(' → '))

# list.extend(): 列表方法，将可迭代对象的元素添加到列表末尾
# 参数: 可迭代对象
# 返回值: 无（修改原列表）

# 统计每个节点的频次
node_freq = Counter(all_nodes)

# 统计相邻节点之间的转换次数（用于识别关键路径）
transitions = []

for path in paths:
    # 拆分路径为节点列表
    steps = path.split(' → ')

    # 提取相邻节点对，如 ('首页', '搜索')
    # range(len(steps)-1): 生成0到倒数第二个索引
    for i in range(len(steps)-1):
        # steps[i]: 当前节点
        # steps[i+1]: 下一个节点
        # 元组 (steps[i], steps[i+1]): 表示一次转换
        transitions.append((steps[i], steps[i+1]))

# 统计转换频次
trans_counts = Counter(transitions)

# 输出Top 10转换路径
print("\nTop 10 节点转换:")

# 元组解包：(src, dst), cnt = (('首页', '搜索'), 182)
# src: 源节点
# dst: 目标节点
# cnt: 转换次数
for (src, dst), cnt in trans_counts.most_common(10):
    print(f"{src} → {dst}: {cnt}次")

# ========== 可视化分析 ==========
"""
图表说明：
- 左图：各节点出现频次的水平条形图，使用viridis渐变色区分
- 右图：用户路径长度分布直方图，红线标注平均步数

业务价值：
- 节点频次反映用户在各功能模块的停留情况
- 路径长度分布反映用户完成目标的效率
"""

# plt.subplots(): 创建图形和子图
# 参数1: 子图行数
# 参数2: 子图列数
# 参数figsize: 图形大小，(宽, 高) 单位英寸
# 返回值: (figure对象, axes数组)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ----- 左图：节点频次条形图 -----

# sorted(): 排序函数
# 参数1: 可迭代对象
# 参数key: 排序键函数，lambda x: x[1] 表示按元组的第2个元素（频次）排序
# 参数reverse: True降序，False升序
# 返回值: 排序后的列表
# node_freq.items(): 返回字典的(键, 值)对列表
sorted_items = sorted(node_freq.items(), key=lambda x: x[1], reverse=True)

# pd.DataFrame(): 创建数据框
# 参数1: 数据（列表的列表或列表的元组）
# 参数columns: 列名列表
node_df = pd.DataFrame(sorted_items, columns=['节点', '次数'])

# axes[0].barh(): 绘制水平柱状图
# 参数1: y轴位置（柱子的垂直位置）
# 参数2: 柱子的长度（宽度）
# 参数color: 柱子颜色，可以是颜色列表
# plt.cm.viridis: matplotlib的viridis色图（渐变色）
# np.linspace(0, 1, n): 生成0到1之间的n个等间距数值
axes[0].barh(range(len(node_df)), node_df['次数'],
             color=plt.cm.viridis(np.linspace(0, 1, len(node_df))))

# axes[0].set_yticks(): 设置y轴刻度位置
axes[0].set_yticks(range(len(node_df)))

# axes[0].set_yticklabels(): 设置y轴刻度标签
axes[0].set_yticklabels(node_df['节点'])

# axes[0].set_title(): 设置子图标题
axes[0].set_title('各节点出现频次')

# axes[0].invert_yaxis(): 反转y轴，让频次最高的显示在顶部
# 参数: 无
# 返回值: 无
axes[0].invert_yaxis()

# ----- 右图：路径长度分布直方图 -----

# 列表推导式：计算每条路径的步数（节点数量）
# len(p.split(' → ')): 拆分后列表的长度
path_lengths = [len(p.split(' → ')) for p in paths]

# axes[1].hist(): 绘制直方图
# 参数1: 数据（数值列表）
# 参数bins: 分箱设置，range(2, 10)表示边界为[2,3,4,5,6,7,8,9]
# 参数color: 柱子颜色
# 参数rwidth: 柱子相对宽度（0-1）
# 参数edgecolor: 柱子边框颜色
# 返回值: (频次数组, 分箱边界, patches对象)
axes[1].hist(path_lengths, bins=range(2, 10), color='#3498db',
             rwidth=0.8, edgecolor='white')

# axes[1].axvline(): 添加垂直参考线
# 参数x: x轴位置
# 参数color: 线条颜色
# 参数linestyle: 线条样式（'--'虚线，'-'实线）
# 参数label: 图例标签
# np.mean(): 计算平均值
axes[1].axvline(np.mean(path_lengths), color='red', linestyle='--',
                label=f'平均{np.mean(path_lengths):.1f}步')

# axes[1].set_title(): 设置标题
axes[1].set_title('用户路径长度分布')

# axes[1].set_xlabel(): 设置x轴标签
axes[1].set_xlabel('路径步数')

# axes[1].set_ylabel(): 设置y轴标签
axes[1].set_ylabel('会话数量')

# axes[1].legend(): 显示图例
axes[1].legend()

# plt.tight_layout(): 自动调整子图间距
plt.tight_layout()

# plt.show(): 显示图形
plt.show()

# ========== 核心洞察 ==========
"""
分析的核心价值：
1. 业务洞察：识别用户主流行为模式，优化产品路径设计
2. 发现的问题：路径过长可能导致用户流失
3. 可落地的建议：
   - 缩短核心路径步数，提升转化效率
   - 优化高频入口（搜索/分类导航）的体验
   - 在关键流失节点添加引导
"""

# f-string格式化输出
print(f"平均路径长度{np.mean(path_lengths):.1f}步，搜索是核心路径入口")