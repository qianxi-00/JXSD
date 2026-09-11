"""
=====================================================================================
文件：12_会话状态.py
对应课案章节：Streamlit → 会话状态（st.session_state）
本节知识点：
  1. 为什么需要它：Streamlit 每次交互都重新执行整个脚本，普通变量会被重置。
  2. 基本用法：`if "key" not in st.session_state: st.session_state.key = 初始值`。
  3. 三个实战：计数器 / 待办清单 / 模拟登录。
  4. `st.rerun()`：手动让脚本立刻重跑一次。
  5. `key` 参数的两个作用：
       ① 用 st.session_state.xxx 直接读写组件值；
       ② 同名组件必须用不同 key 区分。
  6. ★★ session_state 的持久边界（课案表格）：什么操作会保留、什么会重置。
  7. 常见模式：初始化、删除 key、清空全部、用 key 做"表单重置"。
  8. st.session_state 与 @st.cache_data / @st.cache_resource 的区别（第 13 节详讲）。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\12_会话状态.py' `
        --server.headless true --server.port 8612 --browser.gatherUsageStats false
=====================================================================================
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="12 会话状态", page_icon="🧠", layout="wide")

st.title("12 会话状态（st.session_state）")
st.caption("对应课案：Streamlit → 会话状态")

# ---------------------------------------------------------------------------
# 一、为什么需要 session_state
# ---------------------------------------------------------------------------
st.header("一、为什么需要 session_state")

st.error(
    "**Streamlit 默认每次交互都重新运行整个脚本，所有普通变量都会被重置。**"
)

col_plain, col_state = st.columns(2)

with col_plain:
    st.subheader("❌ 用普通变量（无效）")
    st.code(
        '''count = 0                     # 每次重跑都会被重置为 0

if st.button("加一"):
    count += 1                # 改的是这一轮脚本里的局部变量

st.write(f"count = {count}")     # 永远显示 0（或者 1，但下次又回到 0）''',
        language="python",
    )
    # 真实演示：用普通变量
    plain_count = 0
    if st.button("加一（普通变量）", key="plain_btn"):
        plain_count += 1
    st.write(f"count = **{plain_count}**")
    st.caption("点多少次都只会显示 1，因为每次重跑 `plain_count` 都被重新初始化成 0，加一的那个结果没有被保存下来。")

with col_state:
    st.subheader("✅ 用 session_state（有效）")
    st.code(
        '''if "count" not in st.session_state:
    st.session_state.count = 0    # 只在第一次运行时初始化

if st.button("加一"):
    st.session_state.count += 1   # 改的是"会话状态"，会一直保留

st.write(f"count = {st.session_state.count}")''',
        language="python",
    )
    # 真实演示：用 session_state
    if "plain_demo_count" not in st.session_state:
        st.session_state.plain_demo_count = 0
    if st.button("加一（session_state）", key="state_btn"):
        st.session_state.plain_demo_count += 1
    st.write(f"count = **{st.session_state.plain_demo_count}**")
    st.caption("点一次加一次，数值在多次重跑之间被保留下来了 —— 这就是 session_state 的作用。")

st.markdown(
    """
**核心原理一句话：**

> `st.session_state` 是**绑在"当前浏览器标签页的会话"上的一个字典**。
> 它是一个**跨脚本重跑**存在的地方，而普通变量只活在"某一次脚本执行"里。

**初始化必须用 `if "key" not in st.session_state` 判断**，
否则每次重跑都会把你的数据重置回初始值 —— 这是新手最常犯的错误。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 二、持久边界（课案表格）
# ---------------------------------------------------------------------------
st.header("二、★★ session_state 的持久边界 ★★")

st.markdown(
    """
**这是本节最重要的一张表**（课案原文）：

| 操作 | session_state |
|---|---|
| 点击按钮、拖动滑块等交互（脚本重跑 / `st.rerun()`） | ✅ **保留** |
| **刷新浏览器（F5）** | ❌ **重置** |
| 关闭标签页后重新打开 | ❌ **重置** |
| 重新执行 `streamlit run app.py` | ❌ **重置** |

**所以 `session_state` 只在「当前标签页的连续交互中」有效，刷新页面就会清空。**
"""
)

st.info(
    "**这对你意味着什么？**\n\n"
    "1. `session_state` 适合存**临时状态**：当前选中的标签页、计数器、草稿、登录状态。\n"
    "2. **绝对不要**把它当数据库用 —— 刷新就没了。\n"
    "3. 需要真正持久化的数据（用户资料、订单）必须写进**文件 / 数据库 / localStorage**。\n"
    "   （本目录的 `15_图书管理系统` 就是用 JSON 文件做持久化的）"
)

st.markdown("**自己验证一下：**")
st.markdown(
    """
1. 点下面的按钮把计数加到 3；
2. 按 **F5 刷新浏览器**（注意：要在浏览器里刷新，不是在编辑器里保存文件）；
3. 你会发现计数**归零了**。
"""
)

if "boundary_demo" not in st.session_state:
    st.session_state.boundary_demo = 0

col_bd1, col_bd2, col_bd3 = st.columns([1, 1, 2])
with col_bd1:
    if st.button("＋1", key="boundary_plus", width="stretch"):
        st.session_state.boundary_demo += 1
with col_bd2:
    if st.button("归零", key="boundary_reset", width="stretch"):
        st.session_state.boundary_demo = 0
with col_bd3:
    st.metric("boundary_demo（点 +1 后按 F5 试试）", st.session_state.boundary_demo)

st.divider()

# ---------------------------------------------------------------------------
# 三、实战 1：计数器
# ---------------------------------------------------------------------------
st.header("三、实战 1：计数器")

st.code(
    '''# === 初始化 ===
if "count" not in st.session_state:
    st.session_state.count = 0

# === 三个按钮操作同一份状态 ===
col1, col2, col3 = st.columns(3)
if col1.button("减一"):
    st.session_state.count -= 1
if col2.button("归零"):
    st.session_state.count = 0
if col3.button("加一"):
    st.session_state.count += 1

st.metric("当前计数", st.session_state.count)''',
    language="python",
)

# 真实运行课案原文的计数器
if "count" not in st.session_state:
    st.session_state.count = 0

col1, col2, col3 = st.columns(3)
if col1.button("减一", key="counter_minus", width="stretch"):
    st.session_state.count -= 1
if col2.button("归零", key="counter_zero", width="stretch"):
    st.session_state.count = 0
if col3.button("加一", key="counter_plus", width="stretch"):
    st.session_state.count += 1

st.metric("当前计数 st.session_state.count", st.session_state.count)

st.markdown(
    """
**为什么 `st.session_state.count -= 1` 能生效？**

因为按钮点击会触发脚本重跑：
这一次重跑里，`if "count" not in st.session_state` 判断为 False（已经存在了），
所以不会覆盖；然后 `st.session_state.count -= 1` 在**旧值**的基础上减一；
最后的 `st.metric` 读到的是新值。

**如果忘了那个 if 判断会怎样？**每次重跑都把 count 重置成 0，计数器永远不动。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 四、实战 2：待办清单
# ---------------------------------------------------------------------------
st.header("四、实战 2：待办清单")

st.code(
    '''# === 初始化 ===
if "todos" not in st.session_state:
    st.session_state.todos = []

new_item = st.text_input("添加新事项", key="new_todo_input")

if st.button("添加"):
    if new_item:
        st.session_state.todos.append(new_item)
        st.rerun()        # 重跑一次，顺便清空输入框

for i, item in enumerate(st.session_state.todos):
    col_text, col_del = st.columns([4, 1])
    col_text.write(f"{i+1}. {item}")
    if col_del.button("删除", key=f"del_{i}"):    # ★ key 必须唯一
        st.session_state.todos.pop(i)
        st.rerun()''',
    language="python",
)

# 真实运行的待办清单
if "todos" not in st.session_state:
    # 初始给两条示例数据
    st.session_state.todos = ["读取课案「会话状态」一节", "动手跑一跑本页的示例"]

new_item = st.text_input("添加新事项", key="new_todo_input", placeholder="输入后按回车或点添加")

col_add, col_clear_all = st.columns([1, 1])
with col_add:
    if st.button("添加", key="todo_add_btn", type="primary", width="stretch"):
        if new_item.strip():
            st.session_state.todos.append(new_item.strip())
            st.rerun()      # 重跑一次，输入框会被重置（因为它的值是这一轮渲染时的值）
with col_clear_all:
    if st.button("清空全部", key="todo_clear_btn", width="stretch"):
        st.session_state.todos = []
        st.rerun()

if st.session_state.todos:
    st.markdown(f"**当前共 {len(st.session_state.todos)} 条：**")
    for i, item in enumerate(st.session_state.todos):
        col_text, col_del = st.columns([6, 1])
        col_text.write(f"{i + 1}. {item}")
        # ★ 每个删除按钮都必须有唯一的 key，否则 Streamlit 会报
        #   DuplicateWidgetID 错误（因为它们的标签都是"删除"）
        if col_del.button("删除", key=f"del_{i}", width="stretch"):
            st.session_state.todos.pop(i)
            st.rerun()
else:
    st.info("待办清单是空的，在上面添加一条试试。")

st.warning(
    "**注意上面的 `key=f\"del_{i}\"`：**\n\n"
    "页面上有多个标签都为「删除」的按钮。如果不给它们不同的 `key`，"
    "Streamlit 无法区分它们，会直接抛异常：\n\n"
    "```\n"
    "StreamlitDuplicateElementId: There are multiple elements with the same "
    "auto-generated ID\n"
    "```\n\n"
    "**规则：循环里创建的组件，key 里必须带上循环变量。**"
)

st.divider()

# ---------------------------------------------------------------------------
# 五、实战 3：模拟登录
# ---------------------------------------------------------------------------
st.header("五、实战 3：模拟登录")

st.code(
    '''# 存储用户登录状态
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# 模拟登录
if not st.session_state.logged_in:
    name = st.text_input("用户名")
    if st.button("登录"):
        st.session_state.logged_in = True
        st.session_state.user_name = name
        st.rerun()
else:
    st.success(f"已登录：{st.session_state.user_name}")
    if st.button("退出登录"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.rerun()''',
    language="python",
)

# 真实运行的模拟登录（用 if/else 控制整块区域的显示，这叫「登录门禁」）
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.logged_in:
    st.subheader("请登录")
    login_name = st.text_input("用户名", key="login_name_input", placeholder="随便输一个名字即可")
    login_pwd = st.text_input("密码", key="login_pwd_input", type="password", placeholder="随便输")

    if st.button("登录", key="login_btn", type="primary"):
        if not login_name.strip():
            st.error("用户名不能为空")
        elif not login_pwd:
            st.error("密码不能为空")
        else:
            # 把登录状态写进 session_state
            st.session_state.logged_in = True
            st.session_state.user_name = login_name.strip()
            # 用 toast 给一个轻量反馈，然后重跑脚本
            st.toast(f"欢迎回来，{st.session_state.user_name}", icon="👋")
            st.rerun()
else:
    st.success(f"已登录：**{st.session_state.user_name}**")
    st.markdown("下面是「登录后才能看到」的内容：")

    with st.container(border=True):
        st.markdown("#### 🔒 会员专区")
        st.write("这里有只有登录用户才能看到的数据：")
        st.dataframe(
            pd.DataFrame(
                {
                    "项目": ["订单记录", "收藏夹", "优惠券"],
                    "数量": [12, 34, 5],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    if st.button("退出登录", key="logout_btn"):
        # 退出时要把相关的状态都清掉，否则会残留
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.rerun()

st.markdown(
    """
**这就是"登录门禁"的标准写法：**用一个 `session_state` 里的布尔值控制
"显示登录表单"还是"显示已登录内容"。

本目录的 `15_图书管理系统/app.py` 用的就是完全一样的模式
（只是把 `logged_in` 换成了 `role`，用来区分管理员和普通用户）。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 六、st.rerun()
# ---------------------------------------------------------------------------
st.header("六、st.rerun()：手动重跑脚本")

st.markdown(
    """
`st.rerun()` 会**立刻中止当前这一轮脚本的执行**，然后从头开始重跑一遍。

**三个典型用途：**

1. **改完状态后立刻刷新界面** —— 本页所有按钮都这么用。
2. **清空输入框** —— 输入框的值是被 session_state 托管的（见下一节），
   重跑之后它会回到"这一轮渲染时的值"，从而实现清空效果。
3. **让对话框关闭** —— 对话框函数不再被调用，就自然关掉了（见 `08_布局.py`）。

**⚠️ 注意：`st.rerun()` 之后的代码不会执行！**

```python
if st.button("保存"):
    st.session_state.saved = True
    st.rerun()                 # ← 从这里"跳回"脚本开头
    st.success("保存成功")      # ← 这一行永远不会执行！
    # 正确做法：在 rerun 之前把提示信息存进 session_state，
    #           在脚本开头读出来显示；或者干脆不要 rerun。
```
"""
)

col_rr1, col_rr2 = st.columns([1, 2])

with col_rr1:
    if st.button("▶ 演示 st.rerun() 清空输入框", width="stretch"):
        # 记录一条"操作日志"，然后重跑
        if "rerun_log" not in st.session_state:
            st.session_state.rerun_log = 0
        st.session_state.rerun_log += 1
        st.rerun()

with col_rr2:
    if "rerun_log" not in st.session_state:
        st.session_state.rerun_log = 0
    st.write(f"`st.rerun()` 被触发了 **{st.session_state.rerun_log}** 次")
    st.caption("每次点击都会让脚本从头再跑一遍，状态通过 session_state 保留下来。")

st.markdown("**正确的「操作后显示提示」模式：**")
st.code(
    '''# ① 在脚本靠前的位置读出"上一次操作的结果"并显示
if st.session_state.get("flash_message"):
    st.success(st.session_state.flash_message)
    st.session_state.flash_message = None      # 显示完就清掉，避免重复显示

# ② 操作发生的地方：先存消息，再 rerun
if st.button("保存"):
    do_save()
    st.session_state.flash_message = "保存成功！"   # 存起来
    st.rerun()                                    # 然后重跑，消息会在顶部显示出来''',
    language="python",
)

if "flash_message" not in st.session_state:
    st.session_state.flash_message = None

# 这就是上面说的"flash message"模式
if st.session_state.flash_message:
    st.success(st.session_state.flash_message)
    st.session_state.flash_message = None

if st.button("▶ 演示 flash message 模式"):
    st.session_state.flash_message = "✅ 操作完成！这条消息是在 rerun 之后、在页面顶部显示出来的。"
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 七、key 参数的两个作用
# ---------------------------------------------------------------------------
st.header("七、key 参数的两个作用")

st.markdown(
    """
课案原文说 `st.text_input` 的 `key` 参数有两个作用。展开来说：

### 作用 1：用 `st.session_state.xxx` 直接读写组件值

写了 `key="new_todo_input"` 之后，
`st.session_state.new_todo_input` 和变量 `new_item` 拿到的是**同一个值**：

```python
new_item = st.text_input("添加新事项", key="new_todo_input")

# 下面这两种写法完全等价
st.write(new_item)                              # 用返回值
st.write(st.session_state.new_todo_input)       # 用 session_state
```

**更重要的是：你可以通过改 `st.session_state.new_todo_input` 来"设置"输入框的值** ——
这是"程序化清空输入框"的标准做法：

```python
if st.button("清空"):
    st.session_state.new_todo_input = ""   # ★ 直接把 key 对应的值设为空
    st.rerun()                             # 重跑后输入框就空了
```

### 作用 2：同名组件必须用不同 key 区分

页面上如果有两个「添加」按钮，Streamlit 无法自动区分它们，会直接报错：

```
StreamlitDuplicateElementId: There are multiple elements with the same auto-generated ID
```

**规则：**
- 循环里创建的组件 → `key=f"item_{i}"`
- 相同标签但不同用途的组件 → 手工给不同的 key
- **key 在整个页面内必须唯一**（包括侧边栏和主区域，它们是同一个命名空间）
"""
)

st.subheader("现场演示：用 session_state 的值控制输入框")

st.code(
    '''name = st.text_input("你的名字", key="name_input")
st.write("现在输入框里的值是：", st.session_state.name_input)

if st.button("一键改成「张三」"):
    st.session_state.name_input = "张三"      # ★ 直接改 key 的值
    st.rerun()                               # 重跑后输入框就会显示"张三"''',
    language="python",
)

name_input = st.text_input("你的名字", key="name_input", placeholder="随便输点什么")
st.write("现在输入框里的值是：", f"`{st.session_state.name_input}`")

col_k1, col_k2 = st.columns(2)
with col_k1:
    if st.button("一键改成「张三」", key="set_name_btn", width="stretch"):
        # 直接修改 key 对应的值 —— 这是"程序化设置组件值"的标准做法
        st.session_state.name_input = "张三"
        st.rerun()
with col_k2:
    if st.button("一键清空", key="clear_name_btn", width="stretch"):
        st.session_state.name_input = ""
        st.rerun()

st.subheader("查看当前的 session_state 全貌")

st.markdown(
    "**调试利器：**把 `dict(st.session_state)` 丢给 `st.json`，"
    "就能在页面上实时看到当前所有的会话状态。"
)
st.code(
    '''st.json(dict(st.session_state))     # 一眼看清当前有哪些 key、值是什么''',
    language="python",
)

with st.expander("🔍 展开查看当前的 st.session_state（实时）", expanded=False):
    # dict() 把 SessionStateProxy 转成普通字典，st.json 才能序列化
    st.json(dict(st.session_state))

st.divider()

# ---------------------------------------------------------------------------
# 八、session_state 常用操作速查
# ---------------------------------------------------------------------------
st.header("八、session_state 常用操作速查")

st.markdown(
    """
```python
# ① 初始化（★ 必须判断，否则每次重跑都被重置）
if "key" not in st.session_state:
    st.session_state.key = 初始值

# ② 读（三种写法等价）
value = st.session_state.key
value = st.session_state["key"]
value = st.session_state.get("key", 默认值)      # 不存在时给默认值，不报错

# ③ 写（会自动创建）
st.session_state.key = 新值
st.session_state["key"] = 新值

# ④ 判断存在
if "key" in st.session_state: ...

# ⑤ 删除单个 key
del st.session_state["key"]
# 或者
st.session_state.pop("key", None)

# ⑥ 清空全部（慎用，会清掉所有组件的状态）
for k in list(st.session_state.keys()):
    del st.session_state[k]

# ⑦ 遍历
for k, v in st.session_state.items():
    print(k, v)

# ⑧ 用回调函数（on_click）操作状态 —— 比在 if 里改更高效
def increment():
    st.session_state.count += 1

st.button("加一", on_click=increment)     # 注意：这里不需要 if
```
"""
)

st.subheader("回调函数写法（进阶）")

st.markdown(
    """
上面的所有例子都是「点按钮 → 重跑 → 在 `if` 里改状态」。
还有一种更高效的写法：**把改状态的逻辑写成回调函数**，传给 `on_click`。

**区别：**用回调时，状态会在**脚本重跑之前**就被改好，
所以页面渲染时读到的已经是新值，不需要 `st.rerun()`。
"""
)

st.code(
    '''def increment():
    """回调函数：在脚本重跑之前执行"""
    st.session_state.callback_count += 1

def reset():
    st.session_state.callback_count = 0

if "callback_count" not in st.session_state:
    st.session_state.callback_count = 0

# 把函数交给 on_click，而不是写 if
st.button("＋1", on_click=increment)
st.button("归零", on_click=reset)

st.metric("callback_count", st.session_state.callback_count)''',
    language="python",
)


def callback_increment():
    """回调函数：加一。会在脚本重跑之前执行。"""
    st.session_state.callback_count += 1


def callback_reset():
    """回调函数：归零。"""
    st.session_state.callback_count = 0


if "callback_count" not in st.session_state:
    st.session_state.callback_count = 0

col_cb1, col_cb2, col_cb3 = st.columns([1, 1, 2])
with col_cb1:
    # 用 on_click 传回调函数（注意：不用写 if）
    st.button("＋1（回调）", on_click=callback_increment, key="cb_plus", width="stretch")
with col_cb2:
    st.button("归零（回调）", on_click=callback_reset, key="cb_reset", width="stretch")
with col_cb3:
    st.metric("callback_count", st.session_state.callback_count)

st.info(
    "**两种写法的取舍：**\n"
    "- `if st.button(...)` 的写法更直观，适合逻辑简单、需要 rerun 的场景。\n"
    "- `on_click=` 回调的写法少一次重跑，适合**只改状态**的场景；"
    "但要注意回调里**不能调用 `st.write` 之类的渲染函数**（那时还没有要渲染的上下文）。"
)

st.divider()

# ---------------------------------------------------------------------------
# 九、和缓存的区别（承上启下）
# ---------------------------------------------------------------------------
st.header("九、session_state 和缓存的区别（承上启下）")

st.markdown(
    """
初学者很容易把 `st.session_state` 和 `@st.cache_data` 搞混。它们解决的是**两个完全不同的问题**：

| | `st.session_state` | `@st.cache_data` / `@st.cache_resource` |
|---|---|---|
| 解决什么 | **状态丢失**：脚本重跑后变量被重置 | **重复计算**：同样的耗时操作被反复执行 |
| 作用范围 | **单个用户的单个标签页会话** | **所有用户、所有会话共享** |
| 生命周期 | 刷新浏览器就没了 | 进程活着就一直在（可设 `ttl` 过期） |
| 存什么 | 用户交互产生的状态（计数器、草稿、登录态） | 数据（DataFrame）、资源（数据库连接、模型） |

**一句话记忆：**
- `session_state` 管的是「**我记得你刚才做了什么**」；
- `cache` 管的是「**这个耗时的活儿我只干一次**」。

下一节 `13_缓存机制.py` 会详细演示缓存。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 十、课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("十、课案原文的完整示例")
st.code(
    '''import streamlit as st

st.title("会话状态演示")

# === 初始化 ===
if "count" not in st.session_state:
    st.session_state.count = 0
if "todos" not in st.session_state:
    st.session_state.todos = []

# === 计数器 ===
st.subheader("计数器")
col1, col2, col3 = st.columns(3)
if col1.button("减一"):
    st.session_state.count -= 1
if col2.button("归零"):
    st.session_state.count = 0
if col3.button("加一"):
    st.session_state.count += 1

st.metric("当前计数", st.session_state.count)

# === 待办事项 ===
st.subheader("待办事项")
new_item = st.text_input("添加新事项", key="new_todo_input")
if st.button("添加"):
    if new_item:
        st.session_state.todos.append(new_item)
        st.rerun()  # 清空输入框

for i, item in enumerate(st.session_state.todos):
    col_text, col_del = st.columns([4, 1])
    col_text.write(f"{i+1}. {item}")
    if col_del.button("删除", key=f"del_{i}"):
        st.session_state.todos.pop(i)
        st.rerun()''',
    language="python",
)

st.code(
    '''# session_state 常用模式：存储用户登录状态
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# 模拟登录
if not st.session_state.logged_in:
    name = st.text_input("用户名")
    if st.button("登录"):
        st.session_state.logged_in = True
        st.session_state.user_name = name
        st.rerun()
else:
    st.success(f"已登录：{st.session_state.user_name}")
    if st.button("退出登录"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.rerun()''',
    language="python",
)

st.success(
    "**本节要点回顾：**\n"
    "1. Streamlit 每次交互都重跑整个脚本，**普通变量会被重置** → 用 `st.session_state` 跨重跑保存状态。\n"
    "2. 初始化必须写 `if \"key\" not in st.session_state:`，否则每次重跑都会重置。\n"
    "3. **持久边界：交互（含 `st.rerun()`）保留；刷新浏览器 / 重开标签页 / 重启服务都会重置。**"
    "它不是数据库。\n"
    "4. `st.rerun()` 之后的代码不会执行 —— 想显示提示就用「存消息 + 重跑 + 开头读取」的模式。\n"
    "5. `key` 的两个作用：**用 session_state 读写组件值**、**区分同名组件**。\n"
    "6. 循环里创建的组件，`key` 必须带循环变量。\n"
    "7. `session_state` 管状态，`cache` 管性能 —— 两者不是一回事。"
)
