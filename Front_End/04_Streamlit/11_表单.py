"""
=====================================================================================
文件：11_表单.py
对应课案章节：Streamlit → 表单
本节知识点：
  1. st.form + st.form_submit_button：批量收集输入，点提交才算一次。
  2. ★ 为什么需要表单：默认每次输入变化都会重跑脚本；表单把 N 次输入合并成 1 次提交。
  3. 参数：key（表单唯一标识）、clear_on_submit（提交后是否清空）、
     enter_to_submit（在输入框里按回车是否触发表单提交）、border。
  4. st.form_submit_button 的 type="primary"（实心主按钮）/ "secondary"（空心次按钮）。
  5. ★★ 铁律：st.form_submit_button 必须写在 with st.form 块【内部】。
  6. 表单里的组件不能触发"即时重跑"（它们只在提交时一次性生效）。
  7. 表单外可以放按钮；表单里不能嵌套表单。
  8. 表单 + 校验 + 提交的完整实战。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\11_表单.py' `
        --server.headless true --server.port 8611 --browser.gatherUsageStats false
=====================================================================================
"""

import re
import time

import streamlit as st

st.set_page_config(page_title="11 表单", page_icon="🧾", layout="wide")

st.title("11 表单")
st.caption("对应课案：Streamlit → 表单")

# ---------------------------------------------------------------------------
# 为什么需要表单
# ---------------------------------------------------------------------------
st.header("一、为什么需要表单")

st.markdown(
    """
回忆第 6 节讲的核心模型：**用户每做一次输入，整个脚本就会重新执行一遍。**

这在"只有一个输入框"时没问题，但如果有 10 个输入框，用户每填一个就重跑一次，
会带来两个后果：

1. **卡顿**：每次输入都要重算一遍后面的所有逻辑。
2. **体验差**：草稿态被频繁提交，比如"用户名还没输完就报'用户名太短'"。

**`st.form` 就是为了解决这个问题：**
它把内部的 N 个输入组件**打包**起来，只有在用户点击**提交按钮**时才把值一次性发回来。
"""
)

col_before, col_after = st.columns(2)

with col_before:
    st.subheader("❌ 不用表单")
    st.code(
        '''# 每输入一个字符都会重跑脚本
name = st.text_input("用户名")
email = st.text_input("邮箱")
age = st.number_input("年龄")

# 所以下面这些代码会被执行很多很多次
st.write(f"你输入的是：{name} / {email} / {age}")''',
        language="python",
    )
    st.caption("输入 10 个字符 → 脚本重跑 10 次")

with col_after:
    st.subheader("✅ 用表单")
    st.code(
        '''# 只有点「提交」时才重跑一次
with st.form("my_form"):
    name = st.text_input("用户名")
    email = st.text_input("邮箱")
    age = st.number_input("年龄")
    submitted = st.form_submit_button("提交")

if submitted:
    st.write(f"你输入的是：{name} / {email} / {age}")''',
        language="python",
    )
    st.caption("输入 10 个字符 → 脚本可能一次都不重跑，点提交才重跑 1 次")

st.divider()

# ---------------------------------------------------------------------------
# 二、课案原文示例：用户注册表单
# ---------------------------------------------------------------------------
st.header("二、课案原文示例：用户注册表单")

st.code(
    '''import streamlit as st

st.title("用户注册表单")

with st.form("register_form", clear_on_submit=True):
    st.subheader("填写注册信息")

    username = st.text_input("用户名 *", placeholder="请输入用户名")
    password = st.text_input("密码 *", type="password")
    email = st.text_input("邮箱")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=1, max_value=120, value=18)
    with col2:
        gender = st.radio("性别", ["男", "女", "其他"], horizontal=True)

    agree = st.checkbox("我已阅读并同意服务条款")

    # form_submit_button 必须在 with form 块内
    submitted = st.form_submit_button("注册", type="primary")

if submitted:
    if not username or not password:
        st.error("用户名和密码为必填项")
    elif not agree:
        st.warning("请先同意服务条款")
    else:
        st.success(f"注册成功！欢迎 {username}")
        st.json({
            "用户名": username,
            "邮箱": email,
            "年龄": age,
            "性别": gender
        })''',
    language="python",
)

st.markdown("**下面是它的完整复现，可以直接填一填试试：**")
st.divider()

# ---------------------------------------------------------------------------
# 真实运行的注册表单（课案原文 + 详细注释）
# ---------------------------------------------------------------------------
with st.form("register_form", clear_on_submit=True):
    # key="register_form" 是这个表单的唯一标识。
    # 如果页面上有两个表单，必须用不同的 key，否则 Streamlit 会报
    # DuplicateWidgetID 错误。
    #
    # clear_on_submit=True：提交成功后自动清空表单里所有输入框
    # （做"注册""新增"这类场景非常合适，省得用户手动清空）。
    st.subheader("填写注册信息")

    # 表单内部的组件，写法和平常完全一样
    username = st.text_input("用户名 *", placeholder="请输入用户名")
    password = st.text_input("密码 *", type="password")   # type="password" 显示成圆点
    email = st.text_input("邮箱")

    # 表单里照样能用 st.columns 做布局
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=1, max_value=120, value=18)
    with col2:
        # horizontal=True 让单选按钮横排显示
        gender = st.radio("性别", ["男", "女", "其他"], horizontal=True)

    agree = st.checkbox("我已阅读并同意服务条款")

    # ★★★ st.form_submit_button 必须写在 with st.form 块【内部】★★★
    # type="primary" 让它显示成实心主按钮（视觉上的主操作）
    submitted = st.form_submit_button("注册", type="primary")

# 提交结果的判断写在 with 块【外面】
# 这是官方推荐的结构：表单负责收集，提交后在外面统一处理
if submitted:
    # 校验逻辑：从上到下依次判断，命中就 return（这里用 if/elif）
    if not username or not password:
        st.error("用户名和密码为必填项")
    elif not agree:
        st.warning("请先同意服务条款")
    else:
        st.success(f"注册成功！欢迎 {username}")
        st.json(
            {
                "用户名": username,
                "邮箱": email,
                "年龄": age,
                "性别": gender,
            }
        )

st.markdown(
    """
**注意 `clear_on_submit=True` 的效果：**提交一次之后，
表单里所有输入框都被清空了（因为表单重新渲染时用的是初始值）。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 三、type="primary" 与 type="secondary"
# ---------------------------------------------------------------------------
st.header("三、按钮的两种视觉样式")

st.markdown(
    """
`st.form_submit_button` 和 `st.button` 都有 `type` 参数：

| 取值 | 外观 | 什么时候用 |
|---|---|---|
| `"secondary"` | 白底 / 空心（**默认**） | 次要操作：取消、重置、返回 |
| `"primary"` | **实心高亮色** | 主操作：提交、保存、确认、注册 |

**一条设计经验：一个区域里主按钮最多一个。**
如果满屏都是实心按钮，用户就分不清「我该点哪个」了。
"""
)

st.code(
    '''st.form_submit_button("注册", type="primary")      # 实心，主操作
st.form_submit_button("取消", type="secondary")    # 空心，次要操作''',
    language="python",
)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    st.caption('type="primary"（实心）')
    # ★ form_submit_button 必须在 with st.form(...) 块内部，
    #   所以这里也给它套一个表单（哪怕只有一个按钮）。
    with st.form("demo_primary_form"):
        st.form_submit_button("保存并提交", type="primary")
with col_btn2:
    st.caption('type="secondary"（默认空心）')
    with st.form("demo_secondary_form"):
        st.form_submit_button("取消", type="secondary")

st.info(
    "**为什么这里也要套 `with st.form(...)`？**\n\n"
    "因为 Streamlit 有硬性规定：`st.form_submit_button()` **只能用在 `st.form()` 内部**，"
    "写在外部会直接抛 `StreamlitAPIException: st.form_submit_button() must be used "
    "inside an st.form()`。\n\n"
    "这正好也是下一节要讲的「铁律」—— 这里先让你亲眼见到它。"
)

st.divider()

# ---------------------------------------------------------------------------
# 四、四个关键参数
# ---------------------------------------------------------------------------
st.header("四、四个关键参数")

st.markdown(
    """
`st.form(key, clear_on_submit=False, *, enter_to_submit=True, border=True, width="stretch")`

| 参数 | 默认 | 作用 |
|---|---|---|
| `key` | 必填 | 表单唯一标识。**页面上多个表单必须用不同的 key** |
| `clear_on_submit` | `False` | 提交后是否清空所有输入框（做"新增"场景很有用） |
| `enter_to_submit` | `True` | 在**单行输入框**里按回车是否触发表单提交。设为 `False` 可禁用 |
| `border` | `True` | 是否给表单画一个边框（视觉上圈出"这是一组"） |
| `width` | `"stretch"` | 宽度 |
"""
)

st.subheader("示例：clear_on_submit 与 enter_to_submit 的对比")

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("**clear_on_submit=True**（提交后自动清空）")
    st.code('st.form("f1", clear_on_submit=True)', language="python")
    with st.form("demo_clear_true", clear_on_submit=True):
        val_a = st.text_input("输入点什么", key="clear_true_input")
        st.form_submit_button("提交并清空", type="primary")
    st.caption("提交后输入框会变空 —— 适合「新增一条记录」这种连续录入的场景。")

with col_c2:
    st.markdown("**clear_on_submit=False**（默认，保留输入）")
    st.code('st.form("f2", clear_on_submit=False)', language="python")
    with st.form("demo_clear_false", clear_on_submit=False):
        val_b = st.text_input("输入点什么", key="clear_false_input")
        st.form_submit_button("提交并保留", type="primary")
    st.caption("提交后输入框内容还在 —— 适合「查询」「筛选」这类场景。")

st.markdown("**enter_to_submit=False：禁止回车提交**")
st.code(
    '''# 默认情况下，在单行输入框里按回车就等于点提交按钮。
# 如果表单里有多个输入框，用户按回车可能会"意外提交"，可以禁用：
with st.form("f3", enter_to_submit=False):
    st.text_input("用户名")
    st.text_input("邮箱")
    st.form_submit_button("提交")     # 现在必须【点击】才能提交''',
    language="python",
)

with st.form("demo_enter", enter_to_submit=False):
    st.text_input("用户名（试试按回车）", key="enter_demo_1")
    st.text_input("邮箱", key="enter_demo_2")
    st.form_submit_button("提交（必须点击）", type="primary")
st.caption("在这个表单里按回车不会提交 —— 你必须真的点按钮。")

st.divider()

# ---------------------------------------------------------------------------
# 五、★★ 铁律：form_submit_button 必须在 form 内部 ★★
# ---------------------------------------------------------------------------
st.header("五、★★ 铁律：提交按钮必须在表单内部 ★★")

st.error(
    "`st.form_submit_button()` **必须写在 `with st.form(...)` 块【内部】**。"
    "写在外部会直接抛异常：\n\n"
    "```\n"
    "StreamlitAPIException: `st.form_submit_button()` must be used inside an `st.form()`.\n"
    "```\n\n"
    "这不是「警告」，是**硬性错误** —— 页面会直接显示红色报错框。"
)

col_ok, col_bad = st.columns(2)

with col_ok:
    st.markdown("**✅ 正确**")
    st.code(
        '''with st.form("ok_form"):
    name = st.text_input("姓名")
    submitted = st.form_submit_button("提交")    # 在块内

if submitted:
    st.write(name)''',
        language="python",
    )

with col_bad:
    st.markdown("**❌ 错误**")
    st.code(
        '''with st.form("bad_form"):
    name = st.text_input("姓名")

# ❌ 在块外 —— 直接抛 StreamlitAPIException
submitted = st.form_submit_button("提交")

# 正确写法：把它缩进到 with 块里面''',
        language="python",
    )

st.subheader("另一个限制：表单里不能嵌套表单")
st.code(
    '''# ❌ 会直接报错：Forms cannot be nested
with st.form("outer"):
    st.text_input("外层输入")
    with st.form("inner"):
        st.text_input("内层输入")

# ✅ 正确做法：一个表单收集所有输入，或者拆成两个平级的表单''',
    language="python",
)

st.subheader("还有一个限制：表单里的组件不能触发即时重跑")
st.markdown(
    """
在表单内部，组件**不会**在值变化时立刻触发页面重跑 —— 这正是表单的意义。
所以：

| 想做的事 | 能不能放在表单里 |
|---|---|
| 收集用户输入 | ✅ 完全可以 |
| 实时预览输入内容 | ❌ 不行（表单里的值只在提交时才发回） |
| 点击按钮后立即执行某个动作 | ❌ 不行（`st.button` 在表单里不生效，必须用 `form_submit_button`） |
| 显示静态内容（图表、表格） | ✅ 可以，但它们在提交前不会更新 |

**结论：**需要"实时联动"的输入**不要**放进表单；需要"一次性提交"的输入**才**放进表单。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 六、实战：一个带完整校验的注册表单
# ---------------------------------------------------------------------------
st.header("六、实战：带完整校验的表单")

st.markdown(
    """
下面这个表单演示真实项目里的标准流程：

**填写 → 提交 → 逐项校验 → 显示所有错误 / 成功入库**
"""
)

# 用一个正则做邮箱校验
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

with st.form("profile_form", clear_on_submit=False):
    st.subheader("个人资料")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        p_name = st.text_input("姓名 *", placeholder="至少 2 个字符")
        p_email = st.text_input("邮箱 *", placeholder="name@example.com")
        p_phone = st.text_input("手机号", placeholder="11 位数字，可留空")

    with col_f2:
        p_city = st.selectbox("城市", ["北京", "上海", "广州", "深圳", "其他"])
        p_level = st.select_slider("熟练度", options=["入门", "初级", "中级", "高级"])
        p_topics = st.multiselect(
            "感兴趣的方向",
            ["HTML", "CSS", "JavaScript", "Streamlit", "数据可视化"],
            default=["Streamlit"],
        )

    p_bio = st.text_area("个人简介", max_chars=200, height=90, placeholder="最多 200 字")
    p_subscribe = st.checkbox("订阅每周技术摘要", value=True)

    profile_submitted = st.form_submit_button("保存资料", type="primary")

if profile_submitted:
    # ---- 校验阶段：把所有错误收集起来，一次性显示 ----
    errors = []

    if len(p_name.strip()) < 2:
        errors.append("姓名至少需要 2 个字符")

    if not EMAIL_RE.match(p_email.strip()):
        errors.append("邮箱格式不正确（正确格式类似 name@example.com）")

    # 手机号：允许留空，但填了就必须是 11 位数字
    if p_phone.strip() and not (p_phone.strip().isdigit() and len(p_phone.strip()) == 11):
        errors.append("手机号必须是 11 位数字（或者留空）")

    if not p_topics:
        errors.append("请至少选择一个感兴趣的方向")

    if errors:
        st.error(f"有 {len(errors)} 个问题需要修正：")
        for idx, message in enumerate(errors, start=1):
            st.write(f"{idx}. {message}")
    else:
        # ---- 通过：模拟一次"保存" ----
        with st.spinner("正在保存…"):
            time.sleep(0.8)      # 模拟写数据库/调接口的耗时

        st.success("资料已保存！")
        st.json(
            {
                "姓名": p_name.strip(),
                "邮箱": p_email.strip(),
                "手机号": p_phone.strip() or "（未填写）",
                "城市": p_city,
                "熟练度": p_level,
                "感兴趣的方向": p_topics,
                "简介": p_bio.strip() or "（未填写）",
                "订阅摘要": p_subscribe,
            }
        )
        st.toast("资料保存成功", icon="💾")

st.divider()

# ---------------------------------------------------------------------------
# 七、表单的三个使用经验
# ---------------------------------------------------------------------------
st.header("七、三条使用经验")

st.markdown(
    """
**经验 1：错误一次性显示，不要"报一个改一个"。**

```python
errors = []
if len(name) < 2:      errors.append("姓名太短")
if not valid_email:    errors.append("邮箱格式错误")
if not topics:         errors.append("请选择方向")

if errors:
    for msg in errors:
        st.write("·", msg)      # 一次性全列出来
```
用户改一遍就能全部改对，而不是改一个提交一次、再发现下一个错。

**经验 2：查询/筛选表单用 `clear_on_submit=False`；新增/注册表单用 `True`。**

- 查询表单：用户提交后往往还要微调条件 → **保留**输入（默认）。
- 新增表单：提交成功后要接着录下一条 → **清空**更方便（`clear_on_submit=True`）。

**经验 3：表单适合"事务性"输入，不适合"探索性"输入。**

- **事务性**（填完点提交，比如注册、下单、保存配置）→ 用 `st.form`。
- **探索性**（拖滑块看图表变化、勾选项实时过滤）→ **不要**用表单，
  直接让组件触发重跑才是对的交互。
"""
)

st.code(
    '''# 事务性：一次填完，一次提交
with st.form("order"):
    product = st.selectbox("商品", products)
    qty = st.number_input("数量", 1, 100, 1)
    address = st.text_area("收货地址")
    if st.form_submit_button("提交订单", type="primary"):
        create_order(product, qty, address)

# 探索性：改一个条件就立刻看到结果（不用表单）
region = st.selectbox("地区", regions)          # ← 一改就重跑
top_n = st.slider("显示前 N 名", 5, 50, 10)      # ← 一拖就重跑
st.bar_chart(df[df.region == region].head(top_n))''',
    language="python",
)

st.divider()

st.success(
    "**本节要点回顾：**\n"
    "1. `st.form` 把 N 个输入打包成**一次提交**，避免每输入一个字就重跑脚本。\n"
    "2. `st.form_submit_button` **必须写在 `with st.form` 块内部**。\n"
    "3. `st.form` **不能嵌套**；表单里的 `st.button` 不触发即时重跑。\n"
    "4. `clear_on_submit=True` 适合「新增/注册」；默认的 `False` 适合「查询/筛选」。\n"
    "5. `type=\"primary\"` 让主操作按钮变实心 —— 一个区域里主按钮最多一个。\n"
    "6. 校验时把错误**收集成列表一次性显示**，比逐个报错体验好得多。"
)
