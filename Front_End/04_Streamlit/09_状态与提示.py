"""
=====================================================================================
文件：09_状态与提示.py
对应课案章节：Streamlit → 状态与提示
本节知识点（课案表格里的 11 个组件，全部覆盖）：
   1. st.success     成功提示（绿色框）
   2. st.info        信息提示（蓝色框）
   3. st.warning     警告提示（黄色框）
   4. st.error       错误提示（红色框）
   5. st.exception   显示异常详情（课案里的"异常捕获"）
   6. st.spinner     加载动画（转圈 + 提示文字）
   7. st.progress    进度条
   8. st.status      状态容器（分阶段显示执行状态）
   9. st.toast       右上角浮动通知
  10. st.balloons    气球动画
  11. st.snow        雪花动画
额外补充：
  12. st.progress 的 text 参数、st.status 的 state 参数
  13. st.toast 的 icon / duration 参数

★ 本文件的一个设计考虑：
  课案里气球/雪花/进度条/加载动画都是**直接执行**的，这会让页面每次加载都要等好几秒、
  而且动画会反复播放。本文件把所有这些"有副作用的演示"都改成【点按钮才触发】，
  这样正常打开页面是秒开的，需要看效果时再点。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\09_状态与提示.py' `
        --server.headless true --server.port 8609 --browser.gatherUsageStats false
=====================================================================================
"""

import time

import streamlit as st

st.set_page_config(page_title="09 状态与提示", page_icon="💬", layout="wide")

st.title("09 状态与提示")
st.caption("对应课案：Streamlit → 状态与提示")

st.markdown(
    """
| 组件 | 代码 | 作用 |
|---|---|---|
| 成功提示 | `st.success("成功！")` | 绿色提示框 |
| 信息提示 | `st.info("提示信息")` | 蓝色提示框 |
| 警告提示 | `st.warning("注意！")` | 黄色警告框 |
| 错误提示 | `st.error("出错了！")` | 红色错误框 |
| 异常捕获 | `st.exception(e)` | 显示异常详情 |
| 加载动画 | `with st.spinner("处理中..."):` | 显示转圈动画 |
| 进度条 | `bar = st.progress(0)` → `bar.progress(50)` | 显示进度 |
| 状态容器 | `with st.status("正在运行..."):` | 阶段执行状态 |
| Toast通知 | `st.toast("操作成功！")` | 右上角浮动通知 |
| 气球动画 | `st.balloons()` | 庆祝气球动画 |
| 雪花动画 | `st.snow()` | 雪花飘落动画 |
"""
)

st.info(
    "**关于本文件的交互方式：**课案里这些组件都是直接执行的。"
    "但因为它们会「每次都播放动画 / 每次都等待」，本文件统一改成**点按钮触发**，"
    "页面本身打开是秒开的。"
)

st.divider()

# ---------------------------------------------------------------------------
# 一、四种消息提示框
# ---------------------------------------------------------------------------
st.header("一、四种消息提示框")

st.markdown(
    """
这四个函数**长得一样，只是颜色和图标不同**，用来表达四种不同级别的信息：

| 函数 | 颜色 | 什么时候用 |
|---|---|---|
| `st.success` | 🟢 绿色 | 操作成功、校验通过、保存完成 |
| `st.info` | 🔵 蓝色 | 中性说明、使用提示、"你知道吗" |
| `st.warning` | 🟡 黄色 | 有风险但还能继续、数据可能不准确、即将过期 |
| `st.error` | 🔴 红色 | 操作失败、校验不通过、不可恢复的问题 |

**它们都返回一个容器对象**，所以可以先占位再填充，也可以放进 `st.empty()` 里替换。
"""
)

st.code(
    '''st.success("操作成功！✓")
st.info("这是一条信息提示")
st.warning("请注意，这可能有风险")
st.error("发生错误！请检查")''',
    language="python",
)

st.success("操作成功！✓")
st.info("这是一条信息提示")
st.warning("请注意，这可能有风险")
st.error("发生错误！请检查")

st.markdown("**它们也支持 Markdown：**")
st.warning(
    "**⚠️ 重要提醒**\n\n"
    "- 这是一个**列表**\n"
    "- 支持 `行内代码` 和 [链接](https://docs.streamlit.io)\n\n"
    "```python\nprint('代码块也能渲染')\n```"
)

st.markdown("**实战：根据校验结果选择用哪一个**")
st.code(
    '''if not username:
    st.error("用户名不能为空")          # 阻止继续
elif len(username) < 2:
    st.warning("用户名太短，建议至少 2 个字符")   # 能继续但提醒
else:
    st.success(f"用户名 {username} 可用")        # 通过''',
    language="python",
)

demo_username = st.text_input("用户名（实时校验演示）", key="msg_demo_username")
if not demo_username:
    st.info("请输入用户名")
elif len(demo_username) < 2:
    st.warning("用户名太短，建议至少 2 个字符")
else:
    st.success(f"用户名「{demo_username}」可用")

st.divider()

# ---------------------------------------------------------------------------
# 二、st.exception 异常显示
# ---------------------------------------------------------------------------
st.header("二、st.exception 显示异常详情")

st.markdown(
    """
`st.exception(exception)` 会把一个**异常对象**完整地渲染出来（包括类型、消息和堆栈）。

**典型用法是配合 `try / except`：**

```python
try:
    result = 1 / 0
except ZeroDivisionError as e:
    st.error("计算出错了")
    st.exception(e)        # 把完整的堆栈显示给开发者看
```

**什么时候用？**开发/调试阶段非常有用（能直接在页面上看到堆栈，不用去翻终端日志）。
**生产环境慎用** —— 堆栈会暴露文件路径和代码结构，存在信息泄露风险。

⚠️ **本页默认不显示异常堆栈**（那看起来像页面报错了）。
勾选下面的复选框才会真的把堆栈渲染出来 —— **这是故意的，让你看到真实效果又不出现在默认视图里。**
"""
)

st.code(
    '''try:
    result = 1 / 0                    # 故意制造一个异常
except ZeroDivisionError as e:
    st.error("计算出错了：除数不能为 0")
    st.exception(e)                   # ★ 显示完整堆栈''',
    language="python",
)

show_exception = st.checkbox(
    "显示 st.exception 的真实渲染效果（会看到一段堆栈信息，这是【故意】的演示）"
)

if show_exception:
    try:
        # 故意制造一个异常来演示
        _ = 1 / 0
    except ZeroDivisionError as exc:
        st.error("计算出错了：除数不能为 0")
        # st.exception 会把完整的堆栈信息渲染到页面上
        st.exception(exc)
else:
    st.caption("（当前未显示异常堆栈。勾选上面的复选框可以看到 st.exception 的真实效果。）")

st.markdown("**配合 st.error 的对比：**")
st.code(
    '''# st.error：只输出一句话，不给堆栈
st.error("计算出错了：除数不能为 0")

# st.exception：输出完整的异常对象和堆栈
st.exception(e)''',
    language="python",
)
st.error("计算出错了：除数不能为 0　←　`st.error` 只给一句话")
st.caption("↑ 这就是 st.error 与 st.exception 的区别")

st.divider()

# ---------------------------------------------------------------------------
# 三、st.spinner 加载动画
# ---------------------------------------------------------------------------
st.header("三、st.spinner 加载动画")

st.markdown(
    """
`with st.spinner("正在加载数据，请稍候..."):` —— 在代码块执行期间，页面上显示一个转圈动画。

**关键点：**

- 它是一个**上下文管理器**（`with` 块），块内的代码执行期间显示动画，执行完自动消失。
- 里面的代码**只执行一次**（就是正常的同步执行），动画只是"视觉遮罩"。
- ⚠️ **不要在 spinner 里放"每次重跑都会执行"的耗时操作**，
  否则用户每点一下按钮都要等一次 —— 这种应该用 `@st.cache_data` 缓存（见第 13 节）。
"""
)

st.code(
    '''with st.spinner("正在加载数据，请稍候..."):
    time.sleep(2)          # 模拟耗时操作
st.success("加载完成！")''',
    language="python",
)

col_spin1, col_spin2 = st.columns(2)

with col_spin1:
    st.subheader("基础用法")
    if st.button("▶ 演示 spinner（等待 1.5 秒）", width="stretch"):
        with st.spinner("正在加载数据，请稍候..."):
            time.sleep(1.5)      # 模拟耗时的加载
        st.success("加载完成！")
    else:
        st.caption("点上面的按钮看转圈动画")

with col_spin2:
    st.subheader("配合缓存才是正确姿势")
    st.code(
        '''@st.cache_data                 # ★ 加缓存
def load_data():
    time.sleep(2)              # 只有第一次会真的等
    return pd.DataFrame(...)

if st.button("加载数据"):
    with st.spinner("首次加载需要 2 秒，后续秒开..."):
        df = load_data()       # 第二次就命中缓存了
    st.dataframe(df)''',
        language="python",
    )
    st.caption("完整的缓存演示见 `13_缓存机制.py`")

st.divider()

# ---------------------------------------------------------------------------
# 四、st.progress 进度条
# ---------------------------------------------------------------------------
st.header("四、st.progress 进度条")

st.markdown(
    """
`bar = st.progress(0)` 创建一个 0~100 的进度条，之后用 `bar.progress(值)` 更新它。

- 值可以是 **0~100 的整数**，也可以是 **0.0~1.0 的浮点数**。
- `text=` 参数可以在进度条上方显示文字（新版特性，比单独用 `st.empty()` 更简洁）。
- 更新时**不会新建组件**，而是**原地更新**那一条。

**课案原文的写法（用 st.empty 显示旁边的文字）：**

```python
progress_bar = st.progress(0)
status_text = st.empty()
for i in range(101):
    progress_bar.progress(i)
    status_text.text(f"进度：{i}%")
    time.sleep(0.02)
```
"""
)

col_prog1, col_prog2 = st.columns(2)

with col_prog1:
    st.subheader("课案原文的写法")
    if st.button("▶ 演示进度条（课案写法）", width="stretch"):
        # 创建一个进度条对象，初始值 0
        progress_bar = st.progress(0)
        # 创建一个占位符，用来显示旁边的文字
        status_text = st.empty()

        # 循环 101 次，从 0% 走到 100%
        for i in range(101):
            progress_bar.progress(i)          # 原地更新进度条
            status_text.text(f"进度：{i}%")    # 原地更新文字
            time.sleep(0.01)                  # 模拟每一步的耗时

        status_text.success("处理完成！")       # 结束后把文字换成成功提示
    else:
        st.caption("点上面的按钮看进度条")

with col_prog2:
    st.subheader("更简洁的新写法（text 参数）")
    st.code(
        '''bar = st.progress(0, text="准备中...")
for i in range(101):
    bar.progress(i, text=f"处理中... {i}%")   # 文字直接写在进度条上
    time.sleep(0.01)
bar.progress(100, text="完成！")''',
        language="python",
    )
    if st.button("▶ 演示新写法", width="stretch"):
        bar = st.progress(0, text="准备中...")
        for i in range(101):
            # 用一个 progress 调用同时更新进度和文字，不需要额外的 st.empty
            bar.progress(i, text=f"处理中... {i}%")
            time.sleep(0.01)
        bar.progress(100, text="完成！")
    else:
        st.caption("两段代码效果几乎一样，但新写法少了一个 st.empty()")

st.markdown("**实战：处理一批文件时显示进度**")
st.code(
    '''files = ["a.csv", "b.csv", "c.csv", "d.csv"]
progress = st.progress(0, text="开始处理…")

for idx, name in enumerate(files, start=1):
    # 真正处理文件的代码
    time.sleep(0.3)
    # 进度 = 已完成数 / 总数
    progress.progress(idx / len(files), text=f"正在处理 {name}（{idx}/{len(files)}）")

progress.progress(1.0, text="全部处理完成 ✅")''',
    language="python",
)
if st.button("▶ 演示「批量处理 + 进度条」", width="stretch"):
    demo_files = ["订单数据.csv", "用户数据.csv", "商品数据.csv", "日志数据.csv"]
    progress = st.progress(0, text="开始处理…")
    for idx, filename in enumerate(demo_files, start=1):
        time.sleep(0.3)
        # progress 接受 0.0~1.0 的浮点数
        progress.progress(idx / len(demo_files), text=f"正在处理 {filename}（{idx}/{len(demo_files)}）")
    progress.progress(1.0, text="全部处理完成 ✅")

st.divider()

# ---------------------------------------------------------------------------
# 五、st.status 状态容器
# ---------------------------------------------------------------------------
st.header("五、st.status 状态容器")

st.markdown(
    """
`with st.status("正在批量处理数据...", expanded=True) as status:`

这是一个**可折叠的"执行日志"区域**，专门用来显示多步骤流程的进展：

- 执行中：标题左侧是**转圈**图标。
- 执行完：调用 `status.update(label="完成", state="complete")` 变成**绿色对勾**，
  并自动折叠起来。
- `state` 的三个取值：`"running"`（默认）/ `"complete"` / `"error"`。
- `expanded=True` 让它执行期间是展开的；完成后自动收起，不占地方。

**和 st.spinner 的区别：**spinner 只显示一个转圈，看不到"做到哪一步了"；
status 能把每一步的日志留在页面上，用户还能展开回看。**复杂流程优先用 st.status。**
"""
)

st.code(
    '''with st.status("正在批量处理数据...", expanded=True) as status:
    st.write("步骤1：读取数据...")
    time.sleep(1)
    st.write("步骤2：清洗数据...")
    time.sleep(1)
    st.write("步骤3：计算指标...")
    time.sleep(1)
    status.update(label="处理完成！", state="complete")''',
    language="python",
)

col_st1, col_st2 = st.columns(2)

with col_st1:
    st.subheader("课案原文的写法")
    if st.button("▶ 演示 st.status（3 秒）", width="stretch"):
        with st.status("正在批量处理数据...", expanded=True) as status:
            st.write("步骤1：读取数据...")
            time.sleep(1)
            st.write("步骤2：清洗数据...")
            time.sleep(1)
            st.write("步骤3：计算指标...")
            time.sleep(1)
            # 更新标题和状态图标
            status.update(label="处理完成！", state="complete")
    else:
        st.caption("点上面的按钮，会看到分步骤的执行日志")

with col_st2:
    st.subheader("三种 state 的效果")
    st.code(
        '''# 成功完成：绿色对勾
status.update(label="处理完成！", state="complete")

# 出错：红色叉
status.update(label="处理失败", state="error")''',
        language="python",
    )
    if st.button("▶ 演示 state=\"error\"", width="stretch"):
        with st.status("正在连接数据库...", expanded=True) as status:
            st.write("正在尝试连接 127.0.0.1:3306 ...")
            time.sleep(1)
            st.write("连接超时，重试第 1 次...")
            time.sleep(1)
            # error 状态会显示红色叉号
            status.update(label="连接失败：超时", state="error")
    else:
        st.caption("点上面的按钮看 error 状态的样子")

st.markdown("**实战：一个真实的多步骤流程**")
st.code(
    '''with st.status("正在生成报表...", expanded=True) as s:
    st.write("① 从数据库读取原始数据")
    df = load_raw_data()

    st.write(f"② 清洗数据（共 {len(df)} 行）")
    df = clean(df)

    st.write("③ 计算汇总指标")
    summary = summarize(df)

    st.write("④ 渲染图表")
    chart = build_chart(summary)

    s.update(label=f"报表生成完成（{len(df)} 行数据）", state="complete")

st.dataframe(summary)''',
    language="python",
)

st.divider()

# ---------------------------------------------------------------------------
# 六、st.toast 浮动通知
# ---------------------------------------------------------------------------
st.header("六、st.toast 浮动通知")

st.markdown(
    """
`st.toast("操作成功！", icon="✅", duration="short")`

- 在**右上角**弹出一个**小方块**，几秒后**自动消失**，不打断用户。
- `icon`：图标（emoji）。
- `duration`：`"short"`（约 4 秒，默认）/ `"long"`（约 10 秒）/ `"infinite"`（不自动消失）。
- ⚠️ **重要：toast 是在"这一次脚本运行"里被调用就会弹出**。
  所以它通常写在"某个操作成功之后"的分支里，
  或者配合 `st.session_state` 记住"刚刚发生了什么"。
"""
)

st.code(
    '''if st.button("发送通知"):
    st.toast("操作成功！", icon="✅")''',
    language="python",
)

col_toast1, col_toast2 = st.columns(2)

with col_toast1:
    st.subheader("基础用法")
    if st.button("🔔 发送成功通知", width="stretch"):
        st.toast("操作成功！", icon="✅")

    if st.button("⚠️ 发送警告通知", width="stretch"):
        st.toast("磁盘空间不足 10%", icon="⚠️")

    if st.button("⏳ 发送长时间通知（10 秒）", width="stretch"):
        st.toast("这是一个会停留 10 秒的通知", icon="⏳", duration="long")

with col_toast2:
    st.subheader("实战：保存成功后提示")
    st.code(
        '''if st.button("保存"):
    save_to_disk(data)
    st.toast("已保存到 data.json", icon="💾")
    st.rerun()''',
        language="python",
    )
    st.caption("toast 很适合「操作完成了但不需要用户确认」的场景。")
    st.markdown(
        "**toast vs st.success 怎么选？**\n\n"
        "| | `st.toast` | `st.success` |\n"
        "|---|---|---|\n"
        "| 位置 | 右上角浮动 | 页面内的绿色框 |\n"
        "| 是否消失 | ✅ 几秒后自动消失 | ❌ 一直留在页面上 |\n"
        "| 适合 | 轻量反馈（已保存、已复制） | 需要用户看到的结果 |"
    )

st.divider()

# ---------------------------------------------------------------------------
# 七、st.balloons 与 st.snow 动画
# ---------------------------------------------------------------------------
st.header("七、st.balloons 与 st.snow 动画")

st.markdown(
    """
这两个函数**没有参数、没有返回值**，调用一次就播放一次动画：

| 函数 | 效果 | 适合场合 |
|---|---|---|
| `st.balloons()` | 一堆气球从下往上飘 | 庆祝：任务完成、部署成功、达成目标 |
| `st.snow()` | 雪花从上往下飘 | 节日氛围、装饰性效果 |

⚠️ **注意：**动画在**每次脚本重跑时都会重新播放**。
所以它们必须写在"触发条件"里面（比如 `if st.button(...)`），
否则你每点一次页面上的任何按钮，都会再放一次气球。
"""
)

st.code(
    '''if st.button("庆祝一下"):
    st.balloons()

if st.button("下雪了"):
    st.snow()''',
    language="python",
)

col_anim1, col_anim2 = st.columns(2)
with col_anim1:
    if st.button("🎈 庆祝一下（balloons）", width="stretch"):
        st.balloons()
with col_anim2:
    if st.button("❄️ 下雪了（snow）", width="stretch"):
        st.snow()

st.markdown("**实战：任务全部完成时庆祝一下**")
st.code(
    '''if st.button("运行全部测试"):
    results = run_tests()

    if all(r.passed for r in results):
        st.success(f"全部 {len(results)} 个测试通过！")
        st.balloons()                  # ★ 只在真的成功时才放气球
    else:
        st.error(f"{sum(not r.passed for r in results)} 个测试失败")''',
    language="python",
)

if st.button("▶ 演示「测试全部通过」的场景", width="stretch"):
    with st.status("正在运行测试...", expanded=True) as s:
        st.write("test_html.py …… 通过")
        time.sleep(0.4)
        st.write("test_css.py ……  通过")
        time.sleep(0.4)
        st.write("test_js.py ……   通过")
        time.sleep(0.4)
        s.update(label="全部测试通过", state="complete")
    st.success("全部 3 个测试通过！")
    st.balloons()      # 庆祝动画

st.divider()

# ---------------------------------------------------------------------------
# 八、课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("课案原文的完整示例")
st.code(
    '''import streamlit as st
import time

st.title("状态组件演示")

# 四种消息提示
st.success("操作成功！✓")
st.info("这是一条信息提示")
st.warning("请注意，这可能有风险")
st.error("发生错误！请检查")

# 加载动画
with st.spinner("正在加载数据，请稍候..."):
    time.sleep(2)
st.success("加载完成！")

# 进度条
st.subheader("进度条")
progress_bar = st.progress(0)
status_text = st.empty()
for i in range(101):
    progress_bar.progress(i)
    status_text.text(f"进度：{i}%")
    time.sleep(0.02)

# 状态容器
with st.status("正在批量处理数据...", expanded=True) as status:
    st.write("步骤1：读取数据...")
    time.sleep(1)
    st.write("步骤2：清洗数据...")
    time.sleep(1)
    st.write("步骤3：计算指标...")
    time.sleep(1)
    status.update(label="处理完成！", state="complete")

# Toast通知
if st.button("发送通知"):
    st.toast("操作成功！", icon="✅")

# 庆祝动画
if st.button("庆祝一下"):
    st.balloons()

if st.button("下雪了"):
    st.snow()''',
    language="python",
)

st.warning(
    "**与课案原文的差异：**课案里 spinner 会无条件等 2 秒、进度条会无条件跑 101 次、"
    "status 会无条件跑 3 秒 —— 意味着**每次页面加载都要等 8 秒以上**。\n\n"
    "本文件把这些都改成了点按钮触发。**这不是「偷懒」，而是正确做法**："
    "演示性的等待不应该拖慢正常使用。"
)

st.success(
    "**本节要点回顾：**\n"
    "1. 四色提示框：`success`（绿）/ `info`（蓝）/ `warning`（黄）/ `error`（红）—— 按严重程度选。\n"
    "2. `st.exception` 显示完整堆栈，**开发时有用，生产环境慎用**。\n"
    "3. `st.spinner` 是「遮罩」，`st.status` 是「分步日志」—— 复杂流程用后者。\n"
    "4. `st.progress` 的 `text` 参数比额外的 `st.empty()` 更简洁。\n"
    "5. `st.toast` 是右上角自动消失的轻量通知；`st.balloons` / `st.snow` 是庆祝动画。\n"
    "6. 所有「有副作用的演示」都应该放在触发条件里，不要无条件执行。"
)
