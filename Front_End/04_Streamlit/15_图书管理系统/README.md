# 图书管理系统（Streamlit 综合实战）

> 对应课案章节：**Streamlit → 综合实战：图书管理系统**
>
> 这是一个**真的能跑起来、功能完整**的图书管理系统：
> 登录 / 权限、图书增删改查、借阅与归还、搜索、统计、JSON 文件持久化 —— 全部实现。

---

## 一、怎么运行

### 1. 启动（使用工作区自带的虚拟环境）

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run `
    'F:\ProGram\Python_Base\Front_End\04_Streamlit\15_图书管理系统\app.py' `
    --server.headless true --server.port 8615 --browser.gatherUsageStats false
```

浏览器打开 <http://localhost:8615>。

> ⚠️ **入口是 `app.py`，不要单独运行 `admin.py` / `user.py`。**
> 它们是被 `st.navigation` 注册的子页面，单独运行只会看到「请先登录」的提示。
> （这也是一道安全设计：不通过入口就拿不到登录状态。）

### 2. 演示账号

| 角色 | 用户名 | 密码 | 能看到什么 |
|---|---|---|---|
| 管理员 | `admin` | `admin123` | 首页、**图书管理**、读者视角 |
| 普通用户 | `user` | `user123` | 首页、**我的图书馆** |
| 普通用户 | `zhangsan` | `123456` | 同上 |

也可以在登录页的「注册」标签里自己注册一个账号。

---

## 二、文件结构

```
15_图书管理系统/
├── app.py              ← 入口：登录页 + 角色分发 + st.navigation
├── admin.py            ← 管理员页面：图书增删改查 / 用户管理 / 借阅记录 / 数据统计 / 系统设置
├── user.py             ← 读者页面：借阅 / 归还 / 借阅记录 / 查询可借书 / 个人中心
├── data.py             ← 数据层：JSON 文件持久化 + 全部业务规则（唯一的数据出入口）
├── data/
│   └── library.json    ← 数据文件（首次运行自动创建，含 8 本书 + 3 个账号）
└── README.md           ← 本文件
```

### 为什么这样分层？

```
        app.py  admin.py  user.py          ← 界面层：只负责"显示"和"收集输入"
                   │
                   ▼
               data.py                      ← 数据层：只负责"数据"和"业务规则"
                   │
                   ▼
        data/library.json                  ← 存储层：JSON 文件
```

- **界面层**调用 `data.py` 的函数，拿到 `(是否成功, 提示信息)`，直接显示给用户。
- **数据层**返回结果而**不抛异常**，所以界面永远不会出现 Traceback。
- 想把 JSON 换成 MySQL？**只改 `data.py` 一个文件**即可，三个界面文件一行都不用动。

---

## 三、功能清单

### 登录 / 权限
- [x] 用户名 + 密码登录，密码用 `hashlib.sha256` 哈希后存储（不存明文）
- [x] 注册新账号（可选管理员 / 普通用户）
- [x] **基于角色显示不同的导航菜单**（管理员 / 普通用户看到的页面不同）
- [x] 退出登录

### 管理员（`admin.py`）
- [x] **图书增删改查**：关键字搜索（书名/作者/ISBN/出版社）、按分类筛选、只看可借
- [x] 新增图书（书名/作者/ISBN/分类/出版社/出版年/总册数/馆藏地）
- [x] 修改图书（改「总册数」时会自动同步「可借册数」，且不允许小于已借出数）
- [x] 删除图书（有册数在借时拒绝删除，并且需要勾选二次确认）
- [x] **用户管理**：列表 / 新增 / 删除（有未归还图书的用户不能删；系统至少保留一个管理员）
- [x] **借阅记录**：全部流水 + 按状态/用户/书名筛选 + 管理员强制归还 + 导出 CSV
- [x] **数据统计**：馆藏分类分布、可借/已借出对比、借出最多的图书 Top 10、
      借阅最多的读者 Top 10、每日借出趋势
- [x] **系统设置**：查看数据文件路径与内容、下载 JSON 备份、输入 `RESET` 重置全部数据

### 普通用户（`user.py`）
- [x] **借阅图书**：下拉选择可借图书（避免手输书名打错）+ 确认借阅
- [x] **归还图书**：下拉选择自己在借的书 + 确认归还
- [x] **借阅记录**：查询任意用户（默认自己）的完整流水，含借出/应还/归还日期与逾期标记，可导出 CSV
- [x] **查询可借书**：搜索 + 分类筛选 + 只看可借 + 表格下方"快速借阅"
- [x] **个人中心**：账号信息、修改密码、我的借阅概览
- [x] **逾期提醒**：有逾期未还的书时，页面顶部弹出红色警示

---

## 四、数据是怎么持久化的

数据保存在 **`data/library.json`**（UTF-8，`ensure_ascii=False`，中文可直接阅读）。

**首次运行时自动创建**，并写入：

- 8 本示例图书（前端开发 / 程序设计 / 计算机基础 / 数据分析 / 计算机网络 五个分类）
- 3 个账号（admin / user / zhangsan）
- 空的借阅记录

**持久化边界（和 `session_state` 的区别）：**

| 操作 | 数据文件 | `st.session_state`（登录态等） |
|---|---|---|
| 页面交互、`st.rerun()` | ✅ 保留 | ✅ 保留 |
| **刷新浏览器（F5）** | ✅ **保留** | ❌ 重置（需要重新登录） |
| 重启 `streamlit run` | ✅ **保留** | ❌ 重置 |

这就是为什么**业务数据放 JSON 文件、界面状态放 session_state**。

### 数据文件结构

```json
{
  "meta":  { "created_at": "2026-09-11", "version": 1, "description": "..." },
  "books": [
    {
      "id": 1,
      "title": "HTML 与 CSS 设计与构建网站",
      "author": "Jon Duckett",
      "isbn": "978-7-115-33345-6",
      "category": "前端开发",
      "publisher": "人民邮电出版社",
      "year": 2013,
      "total": 5,          // 总册数
      "available": 5,      // 可借册数（借书 -1，还书 +1）
      "place": "佛山图书馆"
    }
  ],
  "users": [
    {
      "username": "admin",
      "password_hash": "……sha256……",
      "name": "系统管理员",
      "role": "admin",           // admin | user
      "created_at": "2026-09-11",
      "borrowed": [1, 2]         // 未归还的【借阅记录 id】列表
    }
  ],
  "records": [
    {
      "id": 1,
      "book_id": 1,
      "book_title": "HTML 与 CSS 设计与构建网站",
      "username": "zhangsan",
      "borrow_date": "2026-09-11",
      "due_date": "2026-10-11",   // 借期 30 天
      "return_date": "",
      "status": "借出"            // 借出 | 已归还
    }
  ],
  "next_book_id": 9,
  "next_record_id": 2
}
```

---

## 五、业务规则（都在 `data.py` 里实现）

### 借书 `data.borrow_book(book_id, username)`

必须**同时**满足以下条件，否则拒绝并返回原因：

1. 用户存在
2. 图书存在
3. 该书的 `available > 0`（还有可借册数）
4. 该用户没有**重复借同一本未归还的书**

成功后**一次性更新三处**，保证数据一致：

- `books[].available -= 1`
- `users[].borrowed` 追加新的记录 id
- `records[]` 新增一条（`status="借出"`，`due_date = 今天 + 30 天`）

### 还书 `data.return_book(book_id, username)`

必须存在一条「该用户借了这本书且未归还」的记录。成功后：

- `records[].status = "已归还"`，写入 `return_date`
- `books[].available += 1`（并且 `min(available, total)`，防御式写法防止越界）
- 从 `users[].borrowed` 里移除该记录 id

### 修改总册数

先算出「当前已借出册数 = total - available」，再让：

```
新 available = 新 total - 已借出册数
```

并且**新 total 不允许小于已借出册数**（否则会出现"可借数为负"的荒谬状态）。

### 删除图书 / 删除用户

- 有册数在借的书**不能删除**
- 有未归还图书的用户**不能删除**
- 系统**至少保留一个管理员**账号

---

## 六、与课案原文的差异（以及为什么）

课案原文的 `data.py` 用的是 **MySQL + SQLModel**：

```python
engine = create_engine("mysql+pymysql://root:密码@localhost:3306/AA", echo=False)

class Book(SQLModel, table=True, ...):
    id: int | None = Field(default=None, primary_key=True)
    ...
```

| 课案原文 | 本实现 | 为什么改 |
|---|---|---|
| MySQL 数据库 | **JSON 文件** | 本项目要求「不依赖数据库」；JSON 文件在任何机器上都能直接跑，不需要装 MySQL、不需要配账号密码 |
| 密码明文写死在连接串里 | 密码用 sha256 哈希存进 JSON | 把密码写进代码是严重的安全问题；哈希存储即使文件泄露也拿不到原密码 |
| `user.borrow_book(title)` 找不到书时会 `book.status = "借出"` 抛 `AttributeError` | `data.borrow_book()` 返回 `(False, "《xxx》已全部借出")` | **数据层不抛异常**，界面层直接把提示显示出来，永远不会出现 Traceback |
| 页面让用户手输书名 | 改成**下拉选择** | 手输书名极容易打错一个字就找不到书；能选择就不要让用户输入 |
| `borrow_history` 只存书名列表 | 存完整的借阅记录表 | 记录里能带上借出日期、应还日期、归还日期、逾期状态，信息量完全不同 |
| 无登录（页面直接就是 admin/user 功能页） | **完整的登录 + 角色权限** | 课案的 `app.py` 只是把两个页面并排；真实系统必须有权限控制 |
| 无增删改查中的"改"和"删" | 完整的**增删改查** | 题目要求功能包含增删改查 |

**课案原文的核心思想完全保留了：**模块化分层（`data.py` 只碰数据，界面文件只碰展示）、
`st.navigation` 多页面、`st.form` 表单收集输入、`st.session_state` 记住登录态。

---

## 七、单独验证数据层

`data.py` 可以直接用 Python 运行，它会跑一遍完整自检（建库 → 登录 → 借书 → 重复借 → 还书 → 重复还）：

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' `
    'F:\ProGram\Python_Base\Front_End\04_Streamlit\15_图书管理系统\data.py'
```

输出示例：

```
[2] 初始统计：
    图书种类: 8
    馆藏总册数: 29
    ...
[5] 借书 / 还书测试（用 zhangsan 借第 1 本书）：
    借书 → True 借阅成功：《HTML 与 CSS 设计与构建网站》，请于 2026-10-11 前归还
    重复借 → False 你已经借了《HTML 与 CSS 设计与构建网站》且尚未归还（应该失败）
    还书 → True 归还成功：《HTML 与 CSS 设计与构建网站》（2026-09-11 借出）
    重复还 → False 你没有借阅《HTML 与 CSS 设计与构建网站》，或已经归还过了（应该失败）

自检完成，没有异常。
```

---

## 八、涉及的知识点索引

| 知识点 | 出现位置 |
|---|---|
| `st.set_page_config`（只在入口调用一次） | `app.py` |
| `st.form` / `st.form_submit_button` | 登录、注册、搜索、新增、修改 |
| `st.session_state`（登录态、查询条件记忆） | `app.py` / `admin.py` / `user.py` |
| `st.navigation` / `st.Page`（**按角色动态生成导航**） | `app.py` |
| `st.stop()`（权限门禁） | `admin.py` / `user.py` |
| `st.tabs` / `st.columns` / `st.container(border=True)` / `st.expander` | 三个界面文件 |
| `st.dataframe` / `st.data_editor` / `st.metric` / `st.json` / `st.column_config` | 三个界面文件 |
| `st.bar_chart` / `st.area_chart` / `st.line_chart` | 首页、数据统计 |
| `st.download_button`（CSV 用 `utf-8-sig` 防乱码） | 借阅记录、数据备份 |
| `st.success/info/warning/error/toast/spinner` | 全局 |
| `st.rerun()`（写操作后立刻刷新） | 所有写操作 |
| `pathlib.Path(__file__).resolve().parent`（不依赖工作目录） | 四个文件都有 |
| `sys.path` 注入（保证 `import data` 成功） | 三个界面文件 |
| `threading.Lock`（保护"读—改—写"三步） | `data.py` |
| `hashlib.sha256` + `hmac.compare_digest` | `data.py` |

---

## 九、常见问题

**Q：刷新浏览器后要重新登录？**
是的。登录状态存在 `st.session_state` 里，按设计刷新就重置。
但**数据（图书、借阅记录）不会丢**，它们在你的操作发生时就已经写进 JSON 文件了。

**Q：数据乱了，想恢复初始状态？**
用管理员登录 → 「图书管理」→ **系统设置** 标签 → 输入 `RESET` → 点「重置为初始数据」。
或者直接删掉 `data/library.json`，下次运行会自动重建。

**Q：端口 8615 被占用？**
换一个端口，例如 `--server.port 8625`。

**Q：能同时开多个浏览器窗口操作吗？**
可以。数据层用 `threading.Lock` 保护了"读—改—写"，不会写坏文件。
但两个窗口的 `session_state`（登录态）是各自独立的。

**Q：为什么 `admin.py` / `user.py` 单独运行只有一行提示？**
因为它们需要 `app.py` 建立的登录态。这是**故意的**：不通过登录入口，
就不应该能直接访问管理页面。这也演示了"权限门禁"这件事。
