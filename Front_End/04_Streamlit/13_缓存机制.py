"""
=====================================================================================
文件：13_缓存机制.py
对应课案章节：Streamlit → 缓存机制
本节知识点：
  1. 为什么要缓存：脚本每次交互都重跑，耗时的操作会被反复执行。
  2. `@st.cache_data`     —— 缓存【数据】（DataFrame / 列表 / 字典等返回副本）
  3. `@st.cache_resource` —— 缓存【资源】（数据库连接、模型、全局单例，返回同一对象）
  4. 两者的核心区别（课案表格 + 深入解释）。
  5. 关键参数：`ttl`（过期时间）、`max_entries`（最多缓存几份）、`show_spinner`。
  6. `函数.clear()` 清空某个函数的缓存、`st.cache_data.clear()` 清空全部。
  7. ★ 怎么证明缓存生效：在函数里"打印一行日志"——第一次会打印，第二次不会。
  8. 缓存的两个经典坑：① 传入不可哈希的参数；② 函数内部有副作用（写文件、改全局变量）。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\13_缓存机制.py' `
        --server.headless true --server.port 8613 --browser.gatherUsageStats false
=====================================================================================
"""

import threading
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="13 缓存机制", page_icon="⚡", layout="wide")

st.title("13 缓存机制")
st.caption("对应课案：Streamlit → 缓存机制")

# ---------------------------------------------------------------------------
# 为什么需要缓存
# ---------------------------------------------------------------------------
st.header("一、为什么需要缓存")

st.markdown(
    """
回忆第 1 节讲的运行模型：**用户每一次操作都会让整个脚本从头重新执行一遍。**

这意味着：如果你在脚本里写了一行「读 100 万行数据 / 调一次接口 / 加载一个模型」，
那么用户每点一次按钮，这个耗时操作就会被**再做一次**。

**`@st.cache_data` / `@st.cache_resource` 就是用来解决这个问题的：**
给函数加上装饰器后，Streamlit 会记住「同样的参数 → 同样的结果」，
第二次调用**直接返回缓存的结果**，不再执行函数体。
"""
)

st.markdown(
    """
| 装饰器 | 用途 | 特点 |
|---|---|---|
| `@st.cache_data` | 缓存**数据**（DataFrame、列表、字典等） | 可设过期时间，支持哈希校验 |
| `@st.cache_resource` | 缓存**资源**（数据库连接、模型加载等） | 不会因参数变化而重新创建 |
"""
)

st.info(
    "**怎么证明缓存真的生效？**本文件在每一个被缓存的函数里都写了"
    "一句 `print(...)`（同时记录调用次数）。\n\n"
    "**注意观察运行 streamlit 的那个终端窗口**：第一次调用会打印「实际执行了计算」，"
    "之后再点按钮就不会再打印了 —— 这就是缓存命中的证据。\n\n"
    "页面上也用计数器和提示框把这件事显示出来了，不用切终端也能看到。"
)

st.divider()

# ---------------------------------------------------------------------------
# 二、@st.cache_data 演示
# ---------------------------------------------------------------------------
st.header("二、@st.cache_data：缓存数据")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(n_rows: int = 100) -> pd.DataFrame:
    """
    模拟一个耗时的数据加载过程（课案原文的写法）。

    参数：
        n_rows  生成多少行数据

    ★ 注意这个函数里的 print 和 time.sleep：
      它们只在【缓存未命中】时执行。第二次用相同参数调用时，
      Streamlit 直接返回上次的结果，函数体根本不会跑。
    """
    # 这三行是"证据"：只要打印出来，就说明函数真的执行了
    print(f"[cache_data] 实际执行了 load_data(n_rows={n_rows})，耗时 2 秒……")
    time.sleep(2)      # 模拟耗时 2 秒的数据库查询 / 文件读取

    # 造一份示例数据：100 天的销售额
    df = pd.DataFrame(
        {
            "日期": pd.date_range("2026-01-01", periods=n_rows, freq="D"),
            # 用一个带点规律的公式，让数据看起来不那么随机
            "销售额": [i * 100 + (i % 7) * 50 for i in range(n_rows)],
            "订单数": [10 + (i % 13) for i in range(n_rows)],
        }
    )
    return df


st.code(
    '''import streamlit as st
import pandas as pd
import time

# 缓存数据加载（只加载一次，后续从缓存读取）
@st.cache_data(ttl=3600)          # ttl=3600 表示 1 小时后缓存过期
def load_data():
    time.sleep(2)                 # 模拟耗时加载
    df = pd.DataFrame({
        "日期": pd.date_range("2026-01-01", periods=100),
        "销售额": [i * 100 + i % 7 * 50 for i in range(100)]
    })
    return df

st.title("缓存演示")

if st.button("加载数据"):
    with st.spinner("首次加载需要 2 秒，后续秒开..."):
        df = load_data()
    st.dataframe(df)
    st.caption("再次点击按钮，数据瞬间显示（命中缓存）")

if st.button("清除缓存"):
    load_data.clear()
    st.success("缓存已清除，下次加载将重新计算")
    st.rerun()''',
    language="python",
)

st.markdown("**下面是它的完整复现。点按钮时注意看「耗时」和「调用来源」：**")

# 记录本页被重跑了几次（用 session_state 跨重跑保存）
if "rerun_count" not in st.session_state:
    st.session_state.rerun_count = 0
st.session_state.rerun_count += 1

# 记录数据被真正加载了几次（这个值只有缓存未命中时才会 +1）
if "real_load_count" not in st.session_state:
    st.session_state.real_load_count = 0

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn1:
    load_clicked = st.button("📥 加载数据", type="primary", width="stretch")
with col_btn2:
    clear_clicked = st.button("🧹 清除 load_data 缓存", width="stretch")
with col_btn3:
    clear_all_clicked = st.button("💥 清除全部缓存", width="stretch")

if clear_clicked:
    # 清空"这一个函数"的缓存
    load_data.clear()
    st.session_state.real_load_count = 0
    st.success("已清除 load_data 的缓存。下次加载会重新计算（又会等 2 秒）。")

if clear_all_clicked:
    # 清空所有 st.cache_data / st.cache_resource 的缓存
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.real_load_count = 0
    st.success("已清除全部缓存（st.cache_data.clear() + st.cache_resource.clear()）。")

if load_clicked:
    data_start = time.perf_counter()
    with st.spinner("正在加载数据…（首次需要 2 秒，之后立刻返回）"):
        df_loaded = load_data(100)
    elapsed = time.perf_counter() - data_start

    # 判断这次是"命中缓存"还是"真的算了一遍"
    # 依据：如果耗时很短（< 0.5 秒），说明没跑那个 time.sleep(2)
    if elapsed < 0.5:
        source = "✅ 命中缓存（函数体没有执行）"
    else:
        source = "⏳ 缓存未命中（函数真的执行了 2 秒的耗时操作）"
        st.session_state.real_load_count += 1

    st.success(f"加载完成，耗时 **{elapsed:.3f} 秒** —— {source}")
    st.dataframe(df_loaded, width="stretch", height=260)
    st.caption("再次点击「加载数据」，数据会瞬间显示（因为命中了缓存）。")

st.markdown(
    f"""
**当前统计：**

| 指标 | 值 | 说明 |
|---|---|---|
| 本页脚本重跑次数 | **{st.session_state.rerun_count}** | 你每点一次任何按钮就 +1 |
| `load_data` 真正执行的次数 | **{st.session_state.real_load_count}** | 只有缓存未命中时才 +1 |
| 本页缓存有效期（ttl） | 3600 秒 | 1 小时不用相同参数调用，缓存就过期 |

**多试几次你会发现：**重跑次数一直在涨，但真正执行的次数几乎不变 ——
这就是缓存的价值。
"""
)

st.markdown("**同时看一下终端里的输出：**")
st.caption(
    "运行本应用的终端里，只有第一次（或清除缓存后）会打印 `[cache_data] 实际执行了 ...`，"
    "之后再点「加载数据」都不会再打印。**这行 print 就是缓存生效的铁证。**"
)

st.divider()

# ---------------------------------------------------------------------------
# 三、@st.cache_resource 演示
# ---------------------------------------------------------------------------
st.header("三、@st.cache_resource：缓存资源")

st.markdown(
    """
`@st.cache_resource` 用于缓存**"不该被复制、应该全局只有一个"的对象**：

- 数据库连接 / 连接池
- 机器学习模型（加载一次要几秒到几分钟）
- 全局配置对象
- 线程池、日志器

**和 `cache_data` 最本质的区别：**

| | `@st.cache_data` | `@st.cache_resource` |
|---|---|---|
| 返回的是 | 数据的**副本**（每次调用返回一个新的等价对象） | **同一个对象**（对象身份 `id()` 相同） |
| 是否可被修改 | ✅ 安全：改了缓存里的副本，不影响别人 | ⚠️ 危险：所有会话共享同一个对象，改了就都改了 |
| 是否要求可序列化 | ✅ 要求（内部用 pickle 做副本） | ❌ 不要求（连接对象通常不能被 pickle） |
| 适合 | DataFrame、列表、字典、计算结果 | 连接、模型、全局单例 |
"""
)

st.subheader("先验证「同一对象」这件事")
st.code(
    '''import threading

@st.cache_resource
def get_lock():
    """缓存一个全局锁对象（真实场景里可能是数据库连接池）。"""
    return threading.Lock()

lock_a = get_lock()
lock_b = get_lock()

# ★ cache_resource 返回的是同一个对象，所以 id 相同
st.write("id(lock_a) == id(lock_b):", id(lock_a) == id(lock_b))    # True

# 对比 cache_data：返回的是副本，id 不同
d1 = load_data(10)
d2 = load_data(10)
st.write("id(d1) == id(d2):", id(d1) == id(d2))                    # False''',
    language="python",
)


@st.cache_resource(show_spinner=False)
def get_shared_lock():
    """
    返回一个全局唯一的锁对象。

    为什么用 cache_resource 而不是 cache_data？
    因为 threading.Lock 是"操作系统级别"的对象，无法被 pickle 序列化，
    用 cache_data 会直接报错。而且它本来就应该是全局唯一的 —— 这正是
    cache_resource 的设计目的。
    """
    print("[cache_resource] 实际执行了 get_shared_lock()：创建了一个新的 Lock 对象")
    return threading.Lock()


@st.cache_resource(show_spinner=False)
def get_fake_db_connection(db_url: str = "sqlite:///demo.db"):
    """
    模拟一个"数据库连接"。

    真实的数据库连接（sqlite3.Connection、sqlalchemy.Engine、
    pymysql.Connection 等）都不能被 pickle，所以必须用 cache_resource。
    """
    print(f"[cache_resource] 实际执行了 get_fake_db_connection({db_url})：建立了连接")
    time.sleep(1.2)      # 模拟建立连接的耗时
    # 用一个简单的字典模拟连接对象
    return {
        "url": db_url,
        "created_at": time.strftime("%H:%M:%S"),
        "type": "模拟连接",
    }


col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("用 @st.cache_resource")
    if st.button("获取共享锁对象 a", key="lock_a_btn", width="stretch"):
        st.session_state.lock_a_id = id(get_shared_lock())
    if st.button("获取共享锁对象 b", key="lock_b_btn", width="stretch"):
        st.session_state.lock_b_id = id(get_shared_lock())

    id_a = st.session_state.get("lock_a_id")
    id_b = st.session_state.get("lock_b_id")

    if id_a and id_b:
        st.write(f"id(a) = `{id_a}`")
        st.write(f"id(b) = `{id_b}`")
        if id_a == id_b:
            st.success("两个 id 完全相同 → 拿到的是**同一个对象**（cache_resource 的特性）")
        else:
            st.warning("两个 id 不同（不应该发生，说明缓存被清了）")
    else:
        st.caption("依次点上面两个按钮，对比两次拿到的对象 id。")

with col_res2:
    st.subheader("用 @st.cache_data 对比")
    if st.button("用 cache_data 取两次数据", key="data_id_btn", width="stretch"):
        d1 = load_data(10)
        d2 = load_data(10)
        st.write(f"id(d1) = `{id(d1)}`")
        st.write(f"id(d2) = `{id(d2)}`")
        if id(d1) == id(d2):
            st.warning("id 相同？（通常不应该）")
        else:
            st.info(
                "两个 id 不同 → **每次拿到的是数据的副本**。\n\n"
                "这正是 cache_data 的安全之处：你改了 d1 不会影响 d2，"
                "也不会影响其它用户拿到的数据。"
            )
    else:
        st.caption("点上面的按钮，对比两次拿到的 DataFrame 的对象 id。")

st.subheader("再看一个更像真实场景的例子：数据库连接")
st.code(
    '''@st.cache_resource
def get_db_connection(db_url="sqlite:///demo.db"):
    # 真实代码：
    # import sqlalchemy
    # engine = sqlalchemy.create_engine(db_url)
    # return engine
    time.sleep(1.2)                       # 模拟建立连接的耗时
    return {"url": db_url, "created_at": time.strftime("%H:%M:%S")}

# 第一次调用会真的建连接（等 1.2 秒），之后都是秒回
conn1 = get_db_connection()
conn2 = get_db_connection()
st.write("连接建立时间:", conn1["created_at"], "（两次相同 → 同一个连接）")''',
    language="python",
)

if st.button("▶ 演示：复用数据库连接", key="db_conn_btn"):
    start = time.perf_counter()
    conn1 = get_fake_db_connection()
    first_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    conn2 = get_fake_db_connection()
    second_elapsed = time.perf_counter() - start

    col_db1, col_db2 = st.columns(2)
    with col_db1:
        st.metric("第一次调用耗时", f"{first_elapsed:.3f} 秒")
        st.caption(f"连接对象：`{conn1}`")
    with col_db2:
        st.metric("第二次调用耗时", f"{second_elapsed:.3f} 秒")
        st.caption(f"连接对象：`{conn2}`")

    if conn1 is conn2:
        st.success(
            "两次拿到的是**同一个连接对象**（`conn1 is conn2` 为 True），"
            "所以第二次几乎不耗时。这就是 cache_resource 的典型用法。"
        )
    st.caption("（清除缓存后再点一次，第一次又会等 1.2 秒。）")

st.divider()

# ---------------------------------------------------------------------------
# 四、两者的完整对比表
# ---------------------------------------------------------------------------
st.header("四、两者的完整对比")

st.markdown(
    """
| 对比项 | `@st.cache_data` | `@st.cache_resource` |
|---|---|---|
| **缓存什么** | 数据：DataFrame、ndarray、list、dict、str、数值 | 资源：连接、模型、锁、客户端、全局配置 |
| **返回什么** | 数据的**副本**（自动 pickle 反序列化） | **同一个对象**（同一个 `id`） |
| **参数要求** | 必须**可哈希**（str/int/float/bool/date/None 等） | 参数只用于"区分不同资源"，不做哈希校验 |
| **能否被修改** | ✅ 安全：改副本不影响缓存 | ⚠️ 危险：改了就影响所有会话 |
| **是否要求可序列化** | ✅ 要求（不可序列化会报错） | ❌ 不要求 |
| **典型例子** | 读 CSV、查数据库返回的表格、API 返回的 JSON | `sqlalchemy.Engine`、`redis.Redis`、`SentenceTransformer` 模型 |
| **共享范围** | **所有用户、所有会话共享** | 同上 |
| **怎么选** | 不确定时**先用它** | 只有"不能复制 / 只能是单例"时才用它 |

**一句话判断：**

> 你返回的东西**能不能被"复制一份"而没有任何问题**？
> 能 → `cache_data`；不能（连接、模型、锁）→ `cache_resource`。
"""
)

st.subheader("选错了会怎样？")

col_wrong1, col_wrong2 = st.columns(2)

with col_wrong1:
    st.markdown("**❌ 用 cache_data 缓存数据库连接**")
    st.code(
        '''@st.cache_data
def get_conn():
    return sqlite3.connect("a.db")

# 报错：TypeError: cannot pickle 'sqlite3.Connection' object
# 因为 cache_data 需要对返回值做 pickle 序列化来生成副本''',
        language="python",
    )

with col_wrong2:
    st.markdown("**❌ 用 cache_resource 缓存 DataFrame**")
    st.code(
        '''@st.cache_resource
def load_df():
    return pd.read_csv("big.csv")

# 不报错，但很危险：
df = load_df()
df.drop(columns=["x"], inplace=True)    # ★ 直接改了【全局共享】的那一份
# → 所有用户、所有会话看到的 df 都被改了！''',
        language="python",
    )

st.warning(
    "第二列的坑非常隐蔽：`cache_resource` 返回的是**同一个对象**，"
    "任何一个会话对它做了原地修改（`inplace=True`、`append`、`sort` 不返回新对象……），"
    "都会影响所有其它用户。**除非确实需要单例，否则一律用 `cache_data`。**"
)

st.divider()

# ---------------------------------------------------------------------------
# 五、常用参数与清除方法
# ---------------------------------------------------------------------------
st.header("五、常用参数与清除方法")

st.markdown(
    """
### 参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `ttl` | 缓存存活时间（秒 / `"1h"` / `timedelta`）。到点自动失效 | 数据会变就用它，比如 `ttl=3600` |
| `max_entries` | 最多缓存多少份结果（超出后按 LRU 淘汰） | 参数组合很多时限制一下，防止内存爆掉 |
| `show_spinner` | 缓存未命中时是否显示转圈动画 | 想自己控制提示就设 `False` |
| `persist` | 是否把缓存持久化到磁盘（跨进程重启保留） | 只对 `cache_data` 有效 |
| `hash_funcs` | 自定义不可哈希参数的处理方式 | 传 ORM 对象 / 自定义类时才需要 |

### 清除缓存

```python
load_data.clear()          # 清空【这一个函数】的缓存
st.cache_data.clear()      # 清空所有 cache_data 缓存
st.cache_resource.clear()  # 清空所有 cache_resource 缓存
```

**什么时候需要手动清？**当缓存依赖的**外部数据变了**、而函数参数没变的时候。
比如数据库里更新了数据，但 `load_data()` 的参数还是 `100`，
Streamlit 会认为"参数没变，缓存还有效"，于是继续返回旧数据。
**这种情况下要么用 `ttl` 自动过期，要么提供一个「刷新」按钮调 `.clear()`。**
"""
)

st.markdown("**实战：给数据加一个「手动刷新」按钮**")
st.code(
    '''with st.sidebar:
    if st.button("🔄 刷新数据"):
        load_data.clear()      # 清掉缓存
        st.rerun()             # 重跑，这次会真的重新执行函数

df = load_data()               # 平时命中缓存；点了刷新之后会重算
st.dataframe(df)''',
    language="python",
)

with st.sidebar:
    st.markdown("---")
    st.subheader("🔄 缓存控制")
    if st.button("刷新数据（清缓存）", width="stretch"):
        load_data.clear()
        st.session_state.real_load_count = 0
        st.rerun()
    if st.button("清空全部缓存", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.real_load_count = 0
        st.rerun()
    st.caption("点「刷新数据」后再点主区域的「加载数据」，会重新等 2 秒。")

st.divider()

# ---------------------------------------------------------------------------
# 六、两个经典坑
# ---------------------------------------------------------------------------
st.header("六、两个经典坑")

st.subheader("坑 1：传了不可哈希的参数")

st.code(
    '''# ❌ 列表、字典、DataFrame、自定义对象都是【不可哈希】的
@st.cache_data
def process(items: list):
    return [i * 2 for i in items]

process([1, 2, 3])
# → UnhashableParamError: Cannot hash argument 'items'


# ✅ 三种解决办法
# 办法 A：在参数名后加下划线 —— 告诉 Streamlit"这个参数不参与缓存键"
@st.cache_data
def process_a(items_):
    return [i * 2 for i in items_]
# ⚠️ 危险：不参与缓存键意味着"不管传什么列表，都返回第一次的结果"！

# 办法 B：转成可哈希的类型（推荐）
@st.cache_data
def process_b(items: tuple):        # 用元组代替列表
    return [i * 2 for i in items]

process_b((1, 2, 3))


# 办法 C：对外层函数做缓存，接收哈希过的 key
@st.cache_data
def load_by_key(key: str):
    return fetch(key)''',
    language="python",
)

st.subheader("坑 2：函数里有副作用")

st.markdown(
    """
被缓存的函数**只在缓存未命中时执行**，所以：

- ❌ `print()` 只在第一次出现（这不是 bug，是本文件的**演示手段**）；
- ❌ 往数据库写数据、改全局变量、写文件 —— **第二次调用不会发生**；
- ❌ 生成随机数 / 读取当前时间 —— **会一直返回第一次的值**。

```python
# ❌ 错误：想每次都记录时间，但缓存后永远是第一次的时间
@st.cache_data
def get_now():
    return time.time()          # 永远返回第一次调用时的时间！

# ✅ 正确：缓存"取数"这个耗时操作，时间戳留在外面
@st.cache_data
def get_data():
    return fetch_from_db()

fetched_at = time.time()        # 在函数外记录时间
data = get_data()
```

**一句话原则：被 `@st.cache_*` 装饰的函数必须是一个"纯函数"——
只根据参数算结果，不产生任何副作用。**
"""
)

st.markdown("**现场验证「缓存会固定第一次的结果」：**")
st.code(
    '''@st.cache_data
def get_cached_time():
    return time.strftime("%H:%M:%S")     # 只会在第一次执行

def get_fresh_time():
    return time.strftime("%H:%M:%S")     # 每次重跑都会执行''',
    language="python",
)


@st.cache_data(show_spinner=False)
def get_cached_time() -> str:
    """被缓存的"取当前时间"—— 它只会返回第一次执行时的时间。"""
    print("[cache_data] 实际执行了 get_cached_time()")
    return time.strftime("%H:%M:%S")


def get_fresh_time() -> str:
    """没被缓存 —— 每次脚本重跑都会重新取一次时间。"""
    return time.strftime("%H:%M:%S")


col_t1, col_t2 = st.columns(2)
with col_t1:
    st.metric("被缓存的时间（永远不变）", get_cached_time())
    st.caption("点任何按钮让页面重跑，这个时间都不会变 —— 因为函数体没执行。")
with col_t2:
    st.metric("未缓存的时间（每次重跑都变）", get_fresh_time())
    st.caption("点任何按钮让页面重跑，这个时间都会更新。")

if st.button("点我重跑一次脚本（观察上面两个时间的差异）"):
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 七、一个真实感的综合例子
# ---------------------------------------------------------------------------
st.header("七、综合例子：数据加载 + 计算 + 模型（模拟）")

st.markdown(
    """
真实的数据应用通常是这样的结构：

```python
# ① 连接类资源：全局一个 → cache_resource
@st.cache_resource
def get_engine():
    return create_engine("postgresql://...")

# ② 数据查询：结果可复制 → cache_data，带 ttl
@st.cache_data(ttl=600, show_spinner="正在查询数据…")
def query_sales(start_date, end_date):
    engine = get_engine()                    # 内部复用同一个连接
    return pd.read_sql(f"SELECT * FROM sales WHERE ...", engine)

# ③ 模型加载：很慢且不能复制 → cache_resource
@st.cache_resource(show_spinner="正在加载模型…")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ④ 页面
@st.cache_data
def compute_metrics(df_hash: str):           # 只传可哈希的参数
    ...
```

**分工总结：**

| 层次 | 用哪个 | 理由 |
|---|---|---|
| 连接 / 模型 / 客户端 | `cache_resource` | 全局单例、不可序列化、创建很慢 |
| 查询结果 / 计算结果 | `cache_data` | 每次要一份独立副本，改了互不影响 |
| 用户交互状态 | `session_state` | 每个用户自己的，不能共享（第 12 节） |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 八、课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("八、课案原文的完整示例")
st.code(
    '''import streamlit as st
import pandas as pd
import time

# 缓存数据加载（只加载一次，后续从缓存读取）
@st.cache_data(ttl=3600)  # ttl=3600 表示1小时后缓存过期
def load_data():
    time.sleep(2)  # 模拟耗时加载
    df = pd.DataFrame({
        "日期": pd.date_range("2026-01-01", periods=100),
        "销售额": [i * 100 + i % 7 * 50 for i in range(100)]
    })
    return df

st.title("缓存演示")

if st.button("加载数据"):
    with st.spinner("首次加载需要2秒，后续秒开..."):  # 页面显示旋转动画 + 提示文字
        df = load_data()
    st.dataframe(df)
    st.caption("再次点击按钮，数据瞬间显示（命中缓存）")

if st.button("清除缓存"):
    load_data.clear()
    st.success("缓存已清除，下次加载将重新计算")
    st.rerun()''',
    language="python",
)

st.success(
    "**本节要点回顾：**\n"
    "1. 缓存解决的是**重复执行**的问题（脚本每次交互都重跑）。\n"
    "2. `@st.cache_data` 缓存**数据**，返回**副本**，要求参数可哈希、返回值可序列化。\n"
    "3. `@st.cache_resource` 缓存**资源**，返回**同一个对象**，适合连接 / 模型 / 锁。\n"
    "4. 不确定用哪个 → **先用 `cache_data`**；只有「不能复制」时才用 `cache_resource`。\n"
    "5. `ttl` 控制过期；`.clear()` 手动清除；外部数据变了要记得刷新。\n"
    "6. 被缓存的函数必须是**纯函数** —— 有副作用（写文件、取当前时间、生成随机数）就会出问题。\n"
    "7. **证明缓存生效的方法：在函数里打印一行日志，看第二次还会不会打印。**"
)
