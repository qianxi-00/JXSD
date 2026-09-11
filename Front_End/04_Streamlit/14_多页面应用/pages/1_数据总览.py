"""
=====================================================================================
文件：14_多页面应用/pages/1_数据总览.py
对应课案章节：Streamlit → 多页面应用 → 方式 1：子页面
本节知识点：
  1. 子页面就是一个普通的 Streamlit 脚本 —— 直接写 st.xxx 即可。
  2. ★★ 子页面【绝对不要】调用 st.set_page_config() ★★
     因为它已经在入口 app.py 里调用过了。
  3. 文件名前面的数字（1_）只用于控制导航顺序。
  4. 子页面里同样可以使用 st.tabs / st.columns / 缓存等所有特性。

运行方式：不需要单独运行本文件，请运行上一层的入口：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\14_多页面应用\\app.py' `
        --server.headless true --server.port 8614 --browser.gatherUsageStats false
=====================================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st

# ★ 这里【没有】st.set_page_config，也不应该有！
#   页面配置由入口脚本 14_多页面应用/app.py 统一负责。

st.title("📊 数据总览")
st.caption("对应课案：Streamlit → 多页面应用 → 方式 1：子页面")

st.markdown(
    """
这是 `pages/` 目录下的第一个子页面。它其实就是一个**普通的 Streamlit 脚本**：
直接调用 `st.xxx`，内容就会渲染在主区域里。
唯一要记住的规矩是：**不要在这里调用 `st.set_page_config()`**。
"""
)

# ---------------------------------------------------------------------------
# 造一份可复现的示例数据
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2026)

# 30 天的销售数据
dates = pd.date_range("2026-01-01", periods=30, freq="D")
sales_df = pd.DataFrame(
    {
        "日期": dates,
        "销售额": (3000 + np.cumsum(rng.normal(0, 400, 30))).round(0),
        "订单数": rng.integers(80, 200, 30),
        "退款额": rng.integers(0, 400, 30),
    }
).set_index("日期")

# 用户数据
users_df = pd.DataFrame(
    {
        "姓名": ["用户A", "用户B", "用户C", "用户D", "用户E", "用户F"],
        "城市": ["北京", "上海", "广州", "深圳", "北京", "杭州"],
        "注册时间": [
            "2026-01-15", "2026-02-20", "2026-03-10",
            "2026-03-28", "2026-04-02", "2026-04-18",
        ],
        "消费金额": [3200, 1580, 4600, 890, 2750, 1980],
    }
)

# ---------------------------------------------------------------------------
# 顶部指标卡
# ---------------------------------------------------------------------------
total_sales = int(sales_df["销售额"].sum())
total_orders = int(sales_df["订单数"].sum())
avg_order = round(total_sales / total_orders, 2)

m1, m2, m3, m4 = st.columns(4)
m1.metric("30 天总销售额", f"¥{total_sales:,}", delta="8.5%")
m2.metric("总订单数", f"{total_orders:,}", delta="12%")
m3.metric("平均单笔金额", f"¥{avg_order}", delta="-3.2%")
m4.metric("活跃用户", f"{len(users_df)}", delta="5.1%")

st.divider()

# ---------------------------------------------------------------------------
# 主内容：用选项卡分成两块（课案原文的写法）
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["销售数据", "用户数据", "明细表"])

with tab1:
    st.subheader("销售趋势")
    st.markdown(
        """
下面是课案原文里的两行代码（`st.line_chart` + `st.dataframe`），
我们把它扩展成了更完整的一组图表。
"""
    )
    st.code(
        '''st.line_chart({"销售额": [120, 200, 150, 300, 280, 350]})
st.dataframe(pd.DataFrame({
    "姓名": ["用户A", "用户B", "用户C"],
    "注册时间": ["2026-01-15", "2026-02-20", "2026-03-10"]
}))''',
        language="python",
    )

    # 课案原文的第一行：直接传字典也能画
    st.caption("① 课案原文写法：st.line_chart 直接吃字典")
    st.line_chart({"销售额": [120, 200, 150, 300, 280, 350]})

    st.caption("② 真实数据：30 天销售额与订单数（注意 x 轴自动用了日期的索引）")
    st.line_chart(sales_df[["销售额", "订单数"]], height=320)

    st.caption("③ 面积图：看各部分的累积构成")
    st.area_chart(sales_df[["销售额"]], height=260)

with tab2:
    st.subheader("用户数据")
    st.caption("④ 课案原文写法：st.dataframe 直接吃字典")

    # 课案原文的第二行：直接传字典
    st.dataframe(
        pd.DataFrame(
            {
                "姓名": ["用户A", "用户B", "用户C"],
                "注册时间": ["2026-01-15", "2026-02-20", "2026-03-10"],
            }
        ),
        width="stretch",
    )

    st.caption("⑤ 更完整的用户表（含分组统计）")
    col_users, col_chart = st.columns([3, 2])

    with col_users:
        st.dataframe(
            users_df,
            width="stretch",
            hide_index=True,
            column_config={
                "消费金额": st.column_config.NumberColumn("消费金额（元）", format="¥%d"),
                "注册时间": st.column_config.TextColumn("注册时间"),
            },
        )

    with col_chart:
        # 按城市分组求和，再用柱状图画出来
        by_city = users_df.groupby("城市", as_index=True)["消费金额"].sum()
        st.bar_chart(by_city)

with tab3:
    st.subheader("原始明细表")
    st.markdown("完整数据 + 下载按钮。")
    st.dataframe(sales_df, width="stretch", height=380)

    # 提供 CSV 下载：注意用 utf-8-sig 让 Excel 打开不乱码
    csv_bytes = sales_df.to_csv().encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载销售明细（CSV）",
        data=csv_bytes,
        file_name="销售明细.csv",
        mime="text/csv",
    )

st.divider()

with st.expander("💡 这个子页面用到了哪些知识点？"):
    st.markdown(
        """
- **多页面应用**：本文件就是 `pages/` 下的一个子页面，由入口脚本 `app.py` 注册并运行。
- **布局**：`st.columns`（指标卡那 4 列、用户表那 2 列）、`st.tabs`（三个选项卡）、
  `st.expander`（本区域）。
- **数据**：`st.dataframe`（含 `column_config` 配置列格式）、`st.metric`（指标卡）、
  `st.download_button`（下载 CSV）。
- **图表**：`st.line_chart` / `st.area_chart` / `st.bar_chart`。
- **pandas**：`groupby` 分组聚合、`set_index` 设置索引、`to_csv` 导出。
"""
    )
