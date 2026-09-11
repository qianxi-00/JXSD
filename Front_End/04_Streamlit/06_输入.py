"""
=====================================================================================
文件：06_输入.py
对应课案章节：Streamlit → 输入
本节知识点（课案表格里的 13 种输入组件，全部覆盖）：
   1. st.button        按钮                → bool
   2. st.text_input    文本输入            → str
   3. st.text_area     文本域              → str
   4. st.number_input  数字输入            → int/float
   5. st.slider        滑块                → int/float
   6. st.select_slider 选择滑块            → 选中值
   7. st.selectbox     下拉选择            → 选中值
   8. st.multiselect   多选                → list
   9. st.checkbox      复选框              → bool
  10. st.radio         单选                → 选中值
  11. st.date_input    日期选择            → date 对象
  12. st.time_input    时间选择            → time 对象
  13. st.file_uploader 文件上传            → UploadedFile
  14. st.color_picker  颜色选择            → str(hex)
  15. st.camera_input  相机输入            → UploadedFile
核心概念（课案原文）：
  ★ Streamlit 的核心交互：用户输入 → 变量获取 → 页面更新。
  ★ 每次输入变化都会让整个脚本重新执行一遍，所以"变量"永远是最新值。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\06_输入.py' `
        --server.headless true --server.port 8606 --browser.gatherUsageStats false
=====================================================================================
"""

import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="06 输入", page_icon="🎛️", layout="wide")

st.title("06 输入组件")
st.caption("对应课案：Streamlit → 输入")

# ---------------------------------------------------------------------------
# 核心概念
# ---------------------------------------------------------------------------
st.header("核心概念：用户输入 → 变量获取 → 页面更新")

st.markdown(
    """
Streamlit 的交互模型非常直白：

```python
name = st.text_input("你的名字")   # ① 用户输入
st.write("你好，" + name)          # ② 变量获取 → ③ 页面更新
```

**当你输入内容时发生了什么？**

1. 浏览器把你输入的值发回服务器；
2. Streamlit **从上到下重新执行整个脚本**；
3. `st.text_input(...)` 这一次**直接返回**你刚才输入的值（不会再等输入）；
4. 后面的 `st.write` 用新值渲染，页面就更新了。

**关键推论：**你不需要写任何"回调函数"来响应变化 ——
**每一行代码都自动是最新的**。这是 Streamlit 最爽的地方，
也是为什么"脚本里的局部变量每次都会被重置"（第 12 节会讲怎么解决）。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 速查表
# ---------------------------------------------------------------------------
st.header("速查表（课案原文）")
st.markdown(
    """
| 组件 | 代码 | 返回值 |
|---|---|---|
| 按钮 | `st.button("点我")` | `bool` |
| 文本输入 | `st.text_input("标签")` | `str` |
| 文本域 | `st.text_area("标签")` | `str` |
| 数字输入 | `st.number_input("标签", min_value=0, max_value=100, value=50)` | `int/float` |
| 滑块 | `st.slider("标签", 0, 100, 50)` | `int/float` |
| 选择滑块 | `st.select_slider("标签", options=["差","中","良","优"])` | 选中值 |
| 下拉选择 | `st.selectbox("标签", ["选项A", "选项B", "选项C"])` | 选中值 |
| 多选 | `st.multiselect("标签", ["A","B","C"], default=["A"])` | `list` |
| 复选框 | `st.checkbox("勾选我")` | `bool` |
| 单选 | `st.radio("标签", ["选项1", "选项2", "选项3"])` | 选中值 |
| 日期选择 | `st.date_input("选择日期")` | `date 对象` |
| 时间选择 | `st.time_input("选择时间")` | `time 对象` |
| 文件上传 | `st.file_uploader("上传文件", type=["csv","png","pdf"])` | `UploadedFile` |
| 颜色选择 | `st.color_picker("选颜色", "#00FFAA")` | `str(hex)` |
| 相机输入 | `st.camera_input("拍照")` | `UploadedFile` |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 一、文本类
# ---------------------------------------------------------------------------
st.header("一、文本类")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.subheader("st.text_input")
    st.markdown(
        """
单行文本输入框，**返回字符串**。

- `label`：标签（必填，会显示在输入框上方）
- `value`：默认值
- `placeholder`：占位提示文字（灰色，输入后消失）
- `type="password"`：变成密码框（输入显示成圆点）
- `max_chars`：最大字符数
- `help`：标签右侧的问号提示
- `key`：组件唯一标识（第 12 节详解）
"""
    )
    name = st.text_input("你的名字", placeholder="请输入...")
    password = st.text_input("密码", type="password", help="这里演示密码框效果")
    if name:
        st.write("你好，" + name)
    else:
        st.caption("（输入名字后这里会显示问候语）")

with col_t2:
    st.subheader("st.text_area")
    st.markdown(
        """
多行文本输入框，**返回字符串**。

- `height`：高度（像素）
- `max_chars`：最大字符数（超出后不能再输入）
- 适合：简介、备注、评论、SQL 语句
"""
    )
    bio = st.text_area("个人简介", max_chars=200, height=100, placeholder="用一句话介绍自己")
    # len() 取字符串长度，用来做字数统计
    st.caption(f"已输入 {len(bio)} / 200 个字符")

st.divider()

# ---------------------------------------------------------------------------
# 二、数字类
# ---------------------------------------------------------------------------
st.header("二、数字类")

col_n1, col_n2, col_n3 = st.columns(3)

with col_n1:
    st.subheader("st.number_input")
    st.markdown(
        """
带加减按钮的数字输入框。

- `min_value` / `max_value`：取值范围
- `value`：默认值
- `step`：每次加减的步长
- ⚠️ **返回 int 还是 float 取决于 `value` 的类型**：
  `value=25` 返回 `int`，`value=25.0` 返回 `float`。
"""
    )
    age = st.number_input("年龄", min_value=1, max_value=120, value=25, step=1)
    st.caption(f"age = {age}（类型：{type(age).__name__}）")

    price = st.number_input("价格", min_value=0.0, value=9.9, step=0.5, format="%.2f")
    st.caption(f"price = {price}（类型：{type(price).__name__}）")

with col_n2:
    st.subheader("st.slider")
    st.markdown(
        """
滑块，适合"在一个区间里连续取值"。

- `st.slider(label, min, max, value, step=)`：数值滑块
- `st.slider(label, min, max, (25, 75))`：**区间滑块**，返回**元组**
- `st.slider(label, 0.0, 10.0, 7.5, step=0.5)`：浮点滑块
"""
    )
    # 数值滑块：0.0~10.0，默认 7.5，步长 0.5
    score = st.slider("评分", 0.0, 10.0, 7.5, step=0.5)
    st.caption(f"score = {score}")

    # 区间滑块：传一个元组作为默认值，返回的也是元组
    price_range = st.slider("价格区间", 0, 1000, (200, 800), step=50)
    st.caption(f"price_range = {price_range}（类型：{type(price_range).__name__}）")
    st.caption(f"下限 = {price_range[0]}，上限 = {price_range[1]}")

with col_n3:
    st.subheader("st.select_slider")
    st.markdown(
        """
"离散"滑块：不是连续数值，而是在**给定的一系列选项**里选。

- `options`：可选项列表
- `value`：默认选中项
- 返回值就是选中的那个元素本身（可以是字符串、数字、甚至元组）
"""
    )
    level = st.select_slider("技能等级", options=["入门", "初级", "中级", "高级", "专家"])
    st.caption(f"level = {level}")

    # 选项也可以是数字
    priority = st.select_slider("优先级", options=[1, 2, 3, 4, 5], value=3)
    st.caption(f"priority = {priority}（类型：{type(priority).__name__}）")

st.divider()

# ---------------------------------------------------------------------------
# 三、选择类
# ---------------------------------------------------------------------------
st.header("三、选择类")

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.subheader("st.selectbox 下拉选择")
    st.markdown(
        """
从一组选项里选**一个**。

- `index`：默认选中第几项（从 0 开始）
- `format_func`：只改变"显示的文字"，返回值不变（很好用！）
- `accept_new_options=True`：允许用户输入列表里没有的新值（1.4x 版本新增）
"""
    )
    city = st.selectbox("城市", ["北京", "上海", "广州", "深圳", "其他"])
    st.caption(f"city = {city}")

    # format_func 示例：选项存的是代码，显示的是中文名
    code = st.selectbox(
        "部门（format_func 演示）",
        ["dev", "ops", "hr"],
        format_func=lambda x: {"dev": "研发部", "ops": "运维部", "hr": "人事部"}[x],
    )
    st.caption(f"实际返回的是代码：code = {code}（但用户看到的是中文）")

    st.subheader("st.multiselect 多选")
    st.markdown(
        """
选**多个**，返回值是 **list**。

- `default`：默认选中的项（列表）
- `max_selections`：最多能选几个
"""
    )
    hobbies = st.multiselect("爱好", ["篮球", "音乐", "编程", "旅游", "摄影"])
    st.caption(f"hobbies = {hobbies}（类型：{type(hobbies).__name__}，共 {len(hobbies)} 项）")

with col_s2:
    st.subheader("st.checkbox 复选框")
    st.markdown(
        """
**单个**勾选框，返回 `bool`。

- 做"开关"用它，比如"启用高级选项"。
- 注意 `value` 是默认勾选状态。
"""
    )
    agree = st.checkbox("我同意服务条款")
    st.caption(f"agree = {agree}（类型：{type(agree).__name__}）")

    advanced = st.checkbox("显示高级选项", value=False)
    if advanced:
        st.info("高级选项已展开（这就是 checkbox 控制显示/隐藏的经典用法）")
        st.slider("高级参数", 0, 100, 50, key="adv_slider")

    st.subheader("st.radio 单选")
    st.markdown(
        """
互斥的单选按钮组。

- `horizontal=True`：让选项**横排**显示（默认是竖排）
- `index`：默认选中第几项
"""
    )
    gender = st.radio("性别", ["男", "女", "其他"], horizontal=True)
    st.caption(f"gender = {gender}")

    st.radio("竖排单选（默认）", ["选项 1", "选项 2", "选项 3"], key="radio_vertical")

st.divider()

# ---------------------------------------------------------------------------
# 四、日期与时间
# ---------------------------------------------------------------------------
st.header("四、日期与时间")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("st.date_input")
    st.markdown(
        """
日期选择器，返回 **`datetime.date`** 对象。

- `min_value` / `max_value`：可选范围
- `value`：默认日期
- 传 `(开始, 结束)` 元组 → **日期区间选择**，返回元组
"""
    )
    birthday = st.date_input(
        "生日",
        value=datetime.date(2000, 1, 1),
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date.today(),
    )
    st.caption(f"birthday = {birthday}（类型：{type(birthday).__name__}）")

    # 日期区间
    date_range = st.date_input(
        "查询区间",
        value=(datetime.date(2026, 1, 1), datetime.date(2026, 1, 31)),
        key="date_range_demo",
    )
    st.caption(f"date_range = {date_range}（长度 {len(date_range) if isinstance(date_range, tuple) else 1}）")

with col_d2:
    st.subheader("st.time_input")
    st.markdown(
        """
时间选择器，返回 **`datetime.time`** 对象。

- `value`：默认时间
- `step`：步长（默认 15 分钟 = `datetime.timedelta(minutes=15)`）
"""
    )
    alarm = st.time_input("提醒时间", value=datetime.time(9, 0))
    st.caption(f"alarm = {alarm}（类型：{type(alarm).__name__}）")

    st.subheader("组合起来：日期 + 时间 = 完整时间戳")
    st.markdown(
        """
Streamlit 没有内置的 `datetime` 选择器，标准做法是
**日期 + 时间两个组件拼起来**：
"""
    )
    st.code(
        '''import datetime

d = st.date_input("日期")
t = st.time_input("时间")
dt = datetime.datetime.combine(d, t)     # ★ 用 combine 拼成 datetime
st.write("你选择的时间是：", dt)''',
        language="python",
    )
    combined = datetime.datetime.combine(birthday, alarm)
    st.write("拼出来的完整时间：", combined)

st.divider()

# ---------------------------------------------------------------------------
# 五、文件 / 颜色 / 相机
# ---------------------------------------------------------------------------
st.header("五、文件、颜色、相机")

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.subheader("st.file_uploader 文件上传")
    st.markdown(
        """
- `type=["csv", "png", "pdf"]`：**限制允许的文件后缀**（强烈建议写，否则用户可能传上来任何东西）
- `accept_multiple_files=True`：允许一次上传多个，返回值变成 list
- 返回值是一个 `UploadedFile` 对象，**像文件一样可以直接读**
- ⚠️ 上传的文件放在**内存**里，脚本重跑后需要重新上传 ——
  要持久化请自己写进磁盘或用 `st.session_state` 保存处理结果。
"""
    )
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded_file is not None:
        st.success(f"已收到文件：{uploaded_file.name}（{uploaded_file.size} 字节）")
        try:
            # pd.read_csv 可以直接吃 UploadedFile 对象（它有 read() 方法）
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(20), width="stretch")
            st.caption(f"共 {len(df)} 行，{len(df.columns)} 列")
        except Exception as exc:      # noqa: BLE001  —— 这里必须兜底，否则坏文件会让页面报错
            st.error(f"这个文件读不出表格：{exc}")
            st.caption("请确认上传的是一个格式正确的 CSV 文件。")
    else:
        st.caption("（没有上传文件时，这里什么都不做 —— 页面不会报错）")

with col_f2:
    st.subheader("st.color_picker 颜色选择")
    st.markdown(
        """
颜色选择器，返回 **`#RRGGBB` 格式的十六进制字符串**。

- 第一个参数是标签，第二个是默认颜色。
- 拿到颜色后通常配合 `st.markdown(..., unsafe_allow_html=True)` 用。
"""
    )
    color = st.color_picker("选一个颜色", "#FF5722")
    st.markdown(
        f"你选的颜色：<span style='color:{color};font-size:22px;'>████</span> "
        f"<code>{color}</code>",
        unsafe_allow_html=True,
    )
    # 用它做一个小型的实时预览
    st.markdown(
        f"""<div style="background:{color};height:60px;border-radius:8px;
        display:flex;align-items:center;justify-content:center;color:#fff;
        font-weight:bold;">实时预览：{color}</div>""",
        unsafe_allow_html=True,
    )

    st.subheader("st.camera_input 相机输入")
    st.markdown(
        """
调用设备摄像头拍照，返回一个 `UploadedFile`（图片）。

- 浏览器会弹出摄像头权限请求；**如果没有摄像头/拒绝授权，组件也会正常显示**，
  只是拍不了照，不会让脚本报错。
- 拍照后可以配合 `st.image()` 显示（见 `07_媒体.py`）。
"""
    )
    camera_photo = st.camera_input("拍一张照片")
    if camera_photo:
        st.image(camera_photo, caption="你拍的照片", width=280)
    else:
        st.caption("（点击上面的组件可以启动摄像头；不拍也不影响页面）")

st.divider()

# ---------------------------------------------------------------------------
# 六、课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("六、课案原文的完整示例")
st.code(
    '''import streamlit as st
import pandas as pd

st.title("输入组件演示")

# 文本类
name = st.text_input("你的名字", placeholder="请输入...")
bio = st.text_area("个人简介", max_chars=200, height=100)

# 数字类
age = st.number_input("年龄", min_value=1, max_value=120, value=25, step=1)
score = st.slider("评分", 0.0, 10.0, 7.5, step=0.5)
level = st.select_slider("技能等级", options=["入门", "初级", "中级", "高级", "专家"])

# 选择类
city = st.selectbox("城市", ["北京", "上海", "广州", "深圳", "其他"])
hobbies = st.multiselect("爱好", ["篮球", "音乐", "编程", "旅游", "摄影"])
agree = st.checkbox("我同意服务条款")
gender = st.radio("性别", ["男", "女", "其他"], horizontal=True)

# 日期时间
import datetime
birthday = st.date_input("生日", min_value=datetime.date(1900, 1, 1))
alarm = st.time_input("提醒时间", value=datetime.time(9, 0))

# 文件
uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

# 颜色
color = st.color_picker("选一个颜色", "#FF5722")
st.markdown(f"你选的颜色：<span style='color:{color}'>████</span> {color}", unsafe_allow_html=True)

# 相机
camera_photo = st.camera_input("拍一张照片")
if camera_photo:
    st.image(camera_photo)''',
    language="python",
)

st.divider()

# ---------------------------------------------------------------------------
# 七、把所有输入汇总起来（"表单实时预览"）
# ---------------------------------------------------------------------------
st.header("七、实战：把所有输入汇总起来")

st.markdown(
    "下面把本页收集到的所有值汇总成一个字典，方便你看清「每个组件的返回值长什么样」。"
)
st.json(
    {
        "文本": {"名字": name, "简介": bio},
        "数字": {"年龄": age, "评分": score, "等级": level},
        "选择": {"城市": city, "爱好": hobbies, "同意条款": agree, "性别": gender},
        "时间": {"生日": str(birthday), "提醒": str(alarm)},
        "颜色": color,
    }
)

st.success(
    "**本节要点回顾：**\n"
    "1. 核心模型：**用户输入 → 变量获取 → 页面更新**，不需要写回调。\n"
    "2. 每个输入组件都有明确的返回值类型 —— 记住类型，就不会写错后续逻辑。\n"
    "3. `st.slider` 传元组就是区间滑块，返回元组。\n"
    "4. `st.file_uploader` 一定要写 `type=` 限制后缀，并且要处理「文件读不出来」的情况。\n"
    "5. `format_func` 可以做到「显示中文、返回代码」，非常实用。"
)
