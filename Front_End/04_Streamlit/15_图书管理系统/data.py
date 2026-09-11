"""
=====================================================================================
文件：15_图书管理系统/data.py
对应课案章节：Streamlit → 综合实战：图书管理系统（数据层）
本文件职责：**唯一的数据访问层**。所有对图书 / 用户 / 借阅记录的读写都经过这里。

本节知识点：
  1. 模块化：把「数据」和「界面」分开写 —— app.py / admin.py / user.py 只负责界面，
     本文件只负责数据。这样界面怎么改都不会弄脏数据逻辑。
  2. 用 **JSON 文件**做持久化：不需要数据库，重启浏览器 / 重启服务数据都还在。
     （课案原文用的是 MySQL + SQLModel，本项目按题目要求改成 JSON 文件持久化，
       这样在任何机器上都能直接跑起来，不需要安装和配置数据库。）
  3. 数据文件放在 `15_图书管理系统/data/library.json`，**首次运行时自动创建并写入种子数据**。
  4. 密码不要明文存储：这里用 `hashlib.sha256` 做一次哈希（教学演示级别；
     生产环境应该用 bcrypt / argon2 这类专门的慢哈希算法）。
  5. 借阅业务的核心规则：
       · 借书：可借册数 > 0，且同一本书同一用户不能重复借
       · 还书：该书必须确实在这个用户的借阅列表里
       · 每次借还都要同时更新「图书的可借册数」和「用户的借阅列表」和「借阅记录」
         —— 这三者必须保持一致，否则数据就乱了
  6. 所有对外函数都返回 `(成功与否, 提示信息)` 或数据本身，
     **绝不抛出异常**给界面层，这样界面永远不会出现 Traceback。

路径约定：所有路径都用 pathlib.Path(__file__).resolve().parent 计算，
         不依赖当前工作目录（这样从任何地方运行都不会找不到文件）。
=====================================================================================
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# =============================================================================
# 一、路径与常量
# =============================================================================
# __file__ 是当前这个 .py 文件的路径；.resolve() 拿到绝对路径；.parent 取所在目录。
# 这样无论从哪里启动 streamlit，都能正确定位到数据文件。
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
LIBRARY_FILE: Path = DATA_DIR / "library.json"

# 借阅期限（天）
BORROW_DAYS: int = 30

# 用一把全局锁保证"读—改—写"这三步不会被并发打断。
# （Streamlit 的每个会话在各自的线程里跑脚本，所以确实存在并发可能。）
_LOCK = threading.Lock()


# =============================================================================
# 二、种子数据：第一次运行时写入的初始内容
# =============================================================================
def _seed_data() -> dict[str, Any]:
    """返回初始数据。每次调用都生成一份新的，避免被外部修改污染。"""
    today = date.today().isoformat()
    return {
        "meta": {
            "created_at": today,
            "version": 1,
            "description": "前端基础课案 —— Streamlit 综合实战：图书管理系统（JSON 文件持久化）",
        },
        "books": [
            {
                "id": 1,
                "title": "HTML 与 CSS 设计与构建网站",
                "author": "Jon Duckett",
                "isbn": "978-7-115-33345-6",
                "category": "前端开发",
                "publisher": "人民邮电出版社",
                "year": 2013,
                "total": 5,
                "available": 5,
                "place": "佛山图书馆",
            },
            {
                "id": 2,
                "title": "JavaScript 高级程序设计",
                "author": "Matt Frisbie",
                "isbn": "978-7-115-54538-1",
                "category": "前端开发",
                "publisher": "人民邮电出版社",
                "year": 2020,
                "total": 4,
                "available": 4,
                "place": "佛山图书馆",
            },
            {
                "id": 3,
                "title": "流畅的 Python",
                "author": "Luciano Ramalho",
                "isbn": "978-7-115-45445-7",
                "category": "程序设计",
                "publisher": "人民邮电出版社",
                "year": 2017,
                "total": 3,
                "available": 3,
                "place": "佛山图书馆",
            },
            {
                "id": 4,
                "title": "深入理解计算机系统",
                "author": "Randal E. Bryant",
                "isbn": "978-7-111-54493-7",
                "category": "计算机基础",
                "publisher": "机械工业出版社",
                "year": 2016,
                "total": 2,
                "available": 2,
                "place": "佛山图书馆",
            },
            {
                "id": 5,
                "title": "数据可视化实战",
                "author": "张三",
                "isbn": "978-7-000-00000-1",
                "category": "数据分析",
                "publisher": "示例出版社",
                "year": 2024,
                "total": 6,
                "available": 6,
                "place": "佛山图书馆",
            },
            {
                "id": 6,
                "title": "Python 数据分析",
                "author": "Wes McKinney",
                "isbn": "978-7-111-60786-7",
                "category": "数据分析",
                "publisher": "机械工业出版社",
                "year": 2018,
                "total": 4,
                "available": 4,
                "place": "佛山图书馆",
            },
            {
                "id": 7,
                "title": "设计模式：可复用面向对象软件的基础",
                "author": "Erich Gamma",
                "isbn": "978-7-111-07575-2",
                "category": "程序设计",
                "publisher": "机械工业出版社",
                "year": 2000,
                "total": 2,
                "available": 2,
                "place": "佛山图书馆",
            },
            {
                "id": 8,
                "title": "图解 HTTP",
                "author": "上野宣",
                "isbn": "978-7-115-35153-5",
                "category": "计算机网络",
                "publisher": "人民邮电出版社",
                "year": 2014,
                "total": 3,
                "available": 3,
                "place": "佛山图书馆",
            },
        ],
        "users": [
            {
                "username": "admin",
                "password_hash": _hash_password("admin123"),
                "name": "系统管理员",
                "role": "admin",
                "created_at": today,
                "borrowed": [],
            },
            {
                "username": "user",
                "password_hash": _hash_password("user123"),
                "name": "普通读者",
                "role": "user",
                "created_at": today,
                "borrowed": [],
            },
            {
                "username": "zhangsan",
                "password_hash": _hash_password("123456"),
                "name": "张三",
                "role": "user",
                "created_at": today,
                "borrowed": [],
            },
        ],
        "records": [],
        "next_book_id": 9,
    }


# =============================================================================
# 三、密码哈希
# =============================================================================
def _hash_password(password: str) -> str:
    """
    把明文密码转成哈希串（不可逆）。

    为什么不能存明文？一旦数据文件泄露，所有用户的密码就直接暴露了。
    哈希之后即使泄露也拿不到原密码。

    ⚠️ 教学说明：这里用的是 sha256（快哈希），**生产环境应该用 bcrypt/argon2**
       这类"故意很慢"的算法，并加随机 salt，才能抵御彩虹表和暴力破解。
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码是否匹配。用 hmac.compare_digest 比较可以避免时序攻击。"""
    import hmac

    return hmac.compare_digest(_hash_password(password), password_hash)


# =============================================================================
# 四、底层读写（load / save）
# =============================================================================
def _ensure_file() -> None:
    """
    确保数据目录和数据文件存在。
    不存在就创建目录并写入种子数据。**首次运行时会自动完成这一步。**
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)      # parents=True 连父目录一起建
    if not LIBRARY_FILE.exists():
        _write_raw(_seed_data())


def _read_raw() -> dict[str, Any]:
    """从磁盘读出原始 JSON 数据。文件损坏时自动回退到种子数据。"""
    _ensure_file()
    try:
        with LIBRARY_FILE.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (json.JSONDecodeError, OSError):
        # 文件被手动改坏了 / 读不了 —— 不抛异常，重建一份种子数据
        data = _seed_data()
        _write_raw(data)
        return data

    # 结构补全：万一某些键缺失（比如手工编辑过），补齐后返回，避免 KeyError
    seed = _seed_data()
    for key, value in seed.items():
        data.setdefault(key, value)
    return data


def _write_raw(data: dict[str, Any]) -> None:
    """把数据写到磁盘。ensure_ascii=False 让中文正常显示（否则会变成 \\uXXXX）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LIBRARY_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_data() -> dict[str, Any]:
    """公开的读取接口（界面层用这个）。"""
    with _LOCK:
        return _read_raw()


def save_data(data: dict[str, Any]) -> None:
    """公开的写入接口。"""
    with _LOCK:
        _write_raw(data)


def reset_data() -> None:
    """把数据重置回种子状态（管理员功能里会用到）。"""
    with _LOCK:
        _write_raw(_seed_data())


def data_file_path() -> Path:
    """返回数据文件路径，界面上展示用。"""
    _ensure_file()
    return LIBRARY_FILE


# =============================================================================
# 五、账号与登录
# =============================================================================
def login(username: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    """
    校验账号密码。

    返回： (是否成功, 提示信息, 用户字典或None)
    注意：**永远不抛异常**，界面层可以直接把提示信息显示出来。
    """
    username = (username or "").strip()
    if not username:
        return False, "请输入用户名", None
    if not password:
        return False, "请输入密码", None

    data = load_data()
    user = _find_user_in(data, username)
    if user is None:
        return False, f"用户「{username}」不存在", None
    if not verify_password(password, user.get("password_hash", "")):
        return False, "密码错误", None

    return True, f"登录成功，欢迎 {user.get('name', username)}", user


def _find_user_in(data: dict[str, Any], username: str) -> dict[str, Any] | None:
    """在已加载的数据里按用户名查找用户（内部函数）。"""
    for user in data.get("users", []):
        if user.get("username") == username:
            return user
    return None


def find_user(username: str) -> dict[str, Any] | None:
    """按用户名查找用户（对外接口）。"""
    data = load_data()
    return _find_user_in(data, username)


def list_users() -> list[dict[str, Any]]:
    """列出所有用户（不含密码哈希，界面上安全展示）。"""
    data = load_data()
    return [
        {
            "用户名": u.get("username", ""),
            "姓名": u.get("name", ""),
            "角色": "管理员" if u.get("role") == "admin" else "普通用户",
            "注册时间": u.get("created_at", ""),
            "在借数量": len(u.get("borrowed", [])),
        }
        for u in data.get("users", [])
    ]


def register_user(
    username: str,
    password: str,
    name: str = "",
    role: str = "user",
) -> tuple[bool, str]:
    """
    注册新用户。

    返回： (是否成功, 提示信息)
    """
    username = (username or "").strip()
    if not username:
        return False, "用户名不能为空"
    if len(username) < 2:
        return False, "用户名至少 2 个字符"
    if not password or len(password) < 6:
        return False, "密码至少 6 位"
    if role not in ("admin", "user"):
        return False, "角色只能是 admin 或 user"

    with _LOCK:
        data = _read_raw()
        if _find_user_in(data, username) is not None:
            return False, f"用户名「{username}」已存在"

        data.setdefault("users", []).append(
            {
                "username": username,
                "password_hash": _hash_password(password),
                "name": (name or username).strip(),
                "role": role,
                "created_at": date.today().isoformat(),
                "borrowed": [],
            }
        )
        _write_raw(data)

    return True, f"已注册用户「{username}」（{'管理员' if role == 'admin' else '普通用户'}）"


def delete_user(username: str) -> tuple[bool, str]:
    """删除用户。如果该用户还有未归还的书，拒绝删除。"""
    username = (username or "").strip()
    with _LOCK:
        data = _read_raw()
        user = _find_user_in(data, username)
        if user is None:
            return False, f"用户「{username}」不存在"
        if user.get("borrowed"):
            return False, f"该用户还有 {len(user['borrowed'])} 本未归还的书，无法删除"
        if user.get("role") == "admin" and sum(
            1 for u in data.get("users", []) if u.get("role") == "admin"
        ) <= 1:
            return False, "系统至少需要保留一个管理员账号"

        data["users"] = [u for u in data.get("users", []) if u.get("username") != username]
        _write_raw(data)
    return True, f"已删除用户「{username}」"


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """修改密码（需要提供原密码）。"""
    if not new_password or len(new_password) < 6:
        return False, "新密码至少 6 位"

    with _LOCK:
        data = _read_raw()
        user = _find_user_in(data, username)
        if user is None:
            return False, f"用户「{username}」不存在"
        if not verify_password(old_password, user.get("password_hash", "")):
            return False, "原密码错误"
        user["password_hash"] = _hash_password(new_password)
        _write_raw(data)
    return True, "密码修改成功"


# =============================================================================
# 六、图书的增删改查
# =============================================================================
def list_books(
    keyword: str = "",
    category: str = "",
    only_available: bool = False,
) -> list[dict[str, Any]]:
    """
    查询图书列表，支持三种过滤条件。

    参数：
        keyword        关键字，匹配 书名 / 作者 / ISBN / 出版社
        category       分类（空字符串表示不筛选）
        only_available 只看还有可借册数的书
    """
    data = load_data()
    keyword = (keyword or "").strip().lower()

    result: list[dict[str, Any]] = []
    for book in data.get("books", []):
        if keyword:
            haystack = " ".join(
                str(book.get(field, ""))
                for field in ("title", "author", "isbn", "publisher", "category")
            ).lower()
            if keyword not in haystack:
                continue
        if category and book.get("category") != category:
            continue
        if only_available and int(book.get("available", 0)) <= 0:
            continue
        result.append(dict(book))     # 返回副本，避免外部误改内存里的数据

    return result


def get_book(book_id: int) -> dict[str, Any] | None:
    """按 id 取一本书。"""
    data = load_data()
    return _find_book_in(data, book_id)


def _find_book_in(data: dict[str, Any], book_id: int) -> dict[str, Any] | None:
    """在已加载的数据里按 id 找书（内部函数）。"""
    for book in data.get("books", []):
        if int(book.get("id", -1)) == int(book_id):
            return book
    return None


def find_book_by_title(title: str) -> dict[str, Any] | None:
    """按书名精确查找（借还书时用）。"""
    title = (title or "").strip()
    if not title:
        return None
    data = load_data()
    for book in data.get("books", []):
        if book.get("title") == title:
            return book
    return None


def add_book(
    title: str,
    author: str,
    isbn: str = "",
    category: str = "未分类",
    publisher: str = "",
    year: int = 0,
    total: int = 1,
    place: str = "佛山图书馆",
) -> tuple[bool, str]:
    """新增图书。返回 (是否成功, 提示信息)。"""
    title = (title or "").strip()
    author = (author or "").strip()

    if not title:
        return False, "书名不能为空"
    if not author:
        return False, "作者不能为空"
    if int(total) < 1:
        return False, "总册数至少为 1"

    with _LOCK:
        data = _read_raw()
        # 同名书不允许重复添加（真实系统里应该用 ISBN 判重，这里两个都检查一下）
        for book in data.get("books", []):
            if book.get("title") == title:
                return False, f"已存在同名图书「{title}」"
        if isbn:
            for book in data.get("books", []):
                if book.get("isbn") and book.get("isbn") == isbn:
                    return False, f"ISBN「{isbn}」已被《{book.get('title')}》占用"

        new_id = int(data.get("next_book_id", 1))
        data.setdefault("books", []).append(
            {
                "id": new_id,
                "title": title,
                "author": author,
                "isbn": isbn.strip(),
                "category": (category or "未分类").strip(),
                "publisher": publisher.strip(),
                "year": int(year) if year else 0,
                "total": int(total),
                "available": int(total),      # 新书全部可借
                "place": (place or "佛山图书馆").strip(),
            }
        )
        data["next_book_id"] = new_id + 1
        _write_raw(data)

    return True, f"已添加《{title}》（{author}），共 {int(total)} 册"


def update_book(book_id: int, **fields: Any) -> tuple[bool, str]:
    """
    修改图书信息。

    可以传任意子集：title / author / isbn / category / publisher / year / total / place

    ★ 一个重要的业务规则：修改「总册数」时要同步调整「可借册数」。
      比如总共 5 册、已借出 2 册（available=3），把总数改成 4，
      那么可借应该变成 4-2=2，而不是保持 3（否则可借数比总数还离谱）。
    """
    with _LOCK:
        data = _read_raw()
        book = _find_book_in(data, book_id)
        if book is None:
            return False, f"id={book_id} 的图书不存在"

        if "title" in fields and not str(fields["title"]).strip():
            return False, "书名不能为空"
        if "author" in fields and not str(fields["author"]).strip():
            return False, "作者不能为空"

        # 处理"总册数"的变化：先算出"当前借出去多少本"
        if "total" in fields:
            new_total = int(fields["total"])
            if new_total < 1:
                return False, "总册数至少为 1"
            borrowed_count = int(book.get("total", 0)) - int(book.get("available", 0))
            if new_total < borrowed_count:
                return False, f"当前有 {borrowed_count} 册已借出，总册数不能少于 {borrowed_count}"
            book["total"] = new_total
            book["available"] = new_total - borrowed_count
            fields.pop("total")

        # 其余字段直接覆盖
        for key, value in fields.items():
            if key in ("id", "available"):
                continue                     # 不允许外部直接改 id 和可借数
            book[key] = int(value) if key == "year" else value

        _write_raw(data)

    return True, f"已更新《{book.get('title')}》的信息"


def delete_book(book_id: int) -> tuple[bool, str]:
    """删除图书。如果还有册数在借，拒绝删除。"""
    with _LOCK:
        data = _read_raw()
        book = _find_book_in(data, book_id)
        if book is None:
            return False, f"id={book_id} 的图书不存在"

        borrowed_count = int(book.get("total", 0)) - int(book.get("available", 0))
        if borrowed_count > 0:
            return False, f"《{book.get('title')}》还有 {borrowed_count} 册在借，无法删除"

        data["books"] = [b for b in data.get("books", []) if int(b.get("id", -1)) != int(book_id)]
        _write_raw(data)

    return True, f"已删除《{book.get('title')}》"


def list_categories() -> list[str]:
    """返回所有出现过的分类（用于下拉筛选）。"""
    data = load_data()
    categories = {str(b.get("category", "")).strip() for b in data.get("books", [])}
    categories.discard("")
    return sorted(categories)


# =============================================================================
# 七、借阅与归还（核心业务）
# =============================================================================
def borrow_book(book_id: int, username: str) -> tuple[bool, str]:
    """
    借书。

    校验规则（任何一条不满足都拒绝）：
      1. 用户存在
      2. 图书存在
      3. 这本书当前可借册数 > 0
      4. 该用户没有重复借同一本书（未归还的情况下）

    成功后要同时更新三处数据：
      · 图书的 available -1
      · 用户的 borrowed 列表 +1
      · 借阅记录里新增一条
    """
    username = (username or "").strip()
    if not username:
        return False, "请先登录"

    with _LOCK:
        data = _read_raw()

        user = _find_user_in(data, username)
        if user is None:
            return False, f"用户「{username}」不存在"

        book = _find_book_in(data, book_id)
        if book is None:
            return False, "图书不存在"

        if int(book.get("available", 0)) <= 0:
            return False, f"《{book.get('title')}》已全部借出，暂无可借册数"

        # user["borrowed"] 里存的是"借阅记录 id"列表，用它来查未归还的书
        records = data.setdefault("records", [])
        already = [
            r for r in records
            if r.get("username") == username
            and int(r.get("book_id", -1)) == int(book_id)
            and r.get("status") == "借出"
        ]
        if already:
            return False, f"你已经借了《{book.get('title')}》且尚未归还"

        today = date.today()
        due = today + timedelta(days=BORROW_DAYS)
        record_id = int(data.get("next_record_id", 1))

        records.append(
            {
                "id": record_id,
                "book_id": int(book_id),
                "book_title": book.get("title", ""),
                "username": username,
                "borrow_date": today.isoformat(),
                "due_date": due.isoformat(),
                "return_date": "",
                "status": "借出",
            }
        )
        data["next_record_id"] = record_id + 1

        # 更新图书可借册数
        book["available"] = int(book.get("available", 0)) - 1
        # 更新用户的借阅列表（存记录 id）
        user.setdefault("borrowed", []).append(record_id)

        _write_raw(data)

    return True, f"借阅成功：《{book.get('title')}》，请于 {due.isoformat()} 前归还"


def return_book(book_id: int, username: str) -> tuple[bool, str]:
    """
    还书。

    校验规则：
      1. 用户存在
      2. 确实有一条"该用户借了这本书且未归还"的记录

    成功后：记录状态改成"已归还"、图书 available +1、用户 borrowed 里移除该记录 id。
    """
    username = (username or "").strip()
    if not username:
        return False, "请先登录"

    with _LOCK:
        data = _read_raw()

        user = _find_user_in(data, username)
        if user is None:
            return False, f"用户「{username}」不存在"

        book = _find_book_in(data, book_id)
        if book is None:
            return False, "图书不存在"

        records = data.get("records", [])
        target = None
        for record in records:
            if (
                record.get("username") == username
                and int(record.get("book_id", -1)) == int(book_id)
                and record.get("status") == "借出"
            ):
                target = record
                break

        if target is None:
            return False, f"你没有借阅《{book.get('title')}》，或已经归还过了"

        target["status"] = "已归还"
        target["return_date"] = date.today().isoformat()

        # 可借册数 +1，但不能超过总册数（防御式写法）
        book["available"] = min(
            int(book.get("available", 0)) + 1,
            int(book.get("total", 0)),
        )

        # 从用户的"在借列表"里移除这条记录 id
        borrowed = user.setdefault("borrowed", [])
        if target.get("id") in borrowed:
            borrowed.remove(target["id"])

        _write_raw(data)

    return True, f"归还成功：《{book.get('title')}》（{target.get('borrow_date')} 借出）"


def borrowed_books(username: str) -> list[dict[str, Any]]:
    """列出某个用户当前【未归还】的书（返回图书字典 + 应还日期）。"""
    data = load_data()
    result: list[dict[str, Any]] = []
    for record in data.get("records", []):
        if record.get("username") != username or record.get("status") != "借出":
            continue
        book = _find_book_in(data, int(record.get("book_id", -1)))
        if book is None:
            continue
        item = dict(book)
        item["borrow_date"] = record.get("borrow_date", "")
        item["due_date"] = record.get("due_date", "")
        item["record_id"] = record.get("id")
        item["overdue"] = _is_overdue(record.get("due_date", ""))
        result.append(item)
    return result


def _is_overdue(due_date: str) -> bool:
    """判断是否已逾期。日期字符串解析失败时按"未逾期"处理（不抛异常）。"""
    try:
        return date.fromisoformat(due_date) < date.today()
    except (ValueError, TypeError):
        return False


def borrow_history(username: str = "", book_id: int | None = None) -> list[dict[str, Any]]:
    """
    查询借阅记录。

    参数：
        username  只查某个用户的（空字符串 = 全部用户）
        book_id   只查某本书的（None = 全部图书）
    """
    data = load_data()
    records = data.get("records", [])

    result: list[dict[str, Any]] = []
    for record in records:
        if username and record.get("username") != username:
            continue
        if book_id is not None and int(record.get("book_id", -1)) != int(book_id):
            continue
        result.append(
            {
                "记录号": record.get("id"),
                "书名": record.get("book_title", ""),
                "用户名": record.get("username", ""),
                "借出日期": record.get("borrow_date", ""),
                "应还日期": record.get("due_date", ""),
                "归还日期": record.get("return_date", "") or "—",
                "状态": record.get("status", ""),
                "是否逾期": "是" if (
                    record.get("status") == "借出" and _is_overdue(record.get("due_date", ""))
                ) else "否",
            }
        )

    # 按记录号倒序（最新的在前面）
    result.sort(key=lambda item: item["记录号"] or 0, reverse=True)
    return result


def list_borrow_records() -> list[dict[str, Any]]:
    """列出全部借阅记录（借阅记录页面用，与 borrow_history() 等价）。"""
    return borrow_history()


def return_by_record(record_id: int) -> tuple[bool, str]:
    """
    按【借阅记录 id】归还（管理员"强制归还"用）。

    为什么需要这个函数？因为界面上用户选中的是"某一条借阅记录"，
    而 return_book() 需要的是 (book_id, username)。
    在这个函数里我们直接拿记录里的信息去调用 return_book()，
    这样业务规则（更新图书可借数、更新用户列表）依然只有一份实现。
    """
    data = load_data()
    for record in data.get("records", []):
        if int(record.get("id", -1)) == int(record_id):
            if record.get("status") != "借出":
                return False, f"记录 {record_id} 已经归还过了"
            return return_book(int(record.get("book_id", -1)), str(record.get("username", "")))
    return False, f"记录 {record_id} 不存在"


# =============================================================================
# 八、统计
# =============================================================================
def stats() -> dict[str, Any]:
    """汇总统计信息，给管理员首页 / 统计页面用。"""
    data = load_data()
    books = data.get("books", [])
    users = data.get("users", [])
    records = data.get("records", [])

    total_copies = sum(int(b.get("total", 0)) for b in books)
    available_copies = sum(int(b.get("available", 0)) for b in books)
    borrowed_copies = total_copies - available_copies

    active_records = [r for r in records if r.get("status") == "借出"]
    overdue_records = [r for r in active_records if _is_overdue(r.get("due_date", ""))]

    return {
        "图书种类": len(books),
        "馆藏总册数": total_copies,
        "可借册数": available_copies,
        "已借出册数": borrowed_copies,
        "用户总数": len(users),
        "管理员数": sum(1 for u in users if u.get("role") == "admin"),
        "借阅记录总数": len(records),
        "当前在借": len(active_records),
        "逾期未还": len(overdue_records),
    }


def books_dataframe_rows(books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    把图书列表转成"适合直接喂给 st.dataframe 的字典列表"（中文列名）。
    界面层调用它就能拿到一张漂亮的中文表格。
    """
    rows: list[dict[str, Any]] = []
    for book in books:
        total = int(book.get("total", 0))
        available = int(book.get("available", 0))
        rows.append(
            {
                "id": book.get("id"),
                "书名": book.get("title", ""),
                "作者": book.get("author", ""),
                "分类": book.get("category", ""),
                "出版社": book.get("publisher", ""),
                "出版年": book.get("year", 0),
                "ISBN": book.get("isbn", ""),
                "馆藏地": book.get("place", ""),
                "总册数": total,
                "可借": available,
                "已借出": total - available,
                "状态": "可借" if available > 0 else "已借完",
            }
        )
    return rows


# =============================================================================
# 九、直接运行本文件时的自检
# =============================================================================
if __name__ == "__main__":
    # 直接执行 `python data.py` 会跑这段自检代码：
    # 创建数据文件、打印数据文件位置、跑一遍借还书流程，验证数据层工作正常。
    print("=" * 70)
    print("data.py 数据层自检")
    print("=" * 70)

    print("\n[1] 数据文件位置：")
    print("   ", data_file_path())

    print("\n[2] 初始统计：")
    for key, value in stats().items():
        print(f"    {key}: {value}")

    print("\n[3] 图书列表（前 3 本）：")
    for row in books_dataframe_rows(list_books())[:3]:
        print(f"    id={row['id']} 《{row['书名']}》 {row['作者']} 可借 {row['可借']}/{row['总册数']}")

    print("\n[4] 登录测试：")
    ok, message, user = login("admin", "admin123")
    print(f"    admin/admin123 → {ok} {message}")
    ok, message, _ = login("admin", "wrong-password")
    print(f"    admin/wrong    → {ok} {message}")

    print("\n[5] 借书 / 还书测试（用 zhangsan 借第 1 本书）：")
    ok, message = borrow_book(1, "zhangsan")
    print(f"    借书 → {ok} {message}")
    ok, message = borrow_book(1, "zhangsan")
    print(f"    重复借 → {ok} {message}（应该失败）")
    ok, message = return_book(1, "zhangsan")
    print(f"    还书 → {ok} {message}")
    ok, message = return_book(1, "zhangsan")
    print(f"    重复还 → {ok} {message}（应该失败）")

    print("\n[6] 借阅记录：")
    for item in borrow_history()[:3]:
        print(f"    记录{item['记录号']} 《{item['书名']}》 {item['用户名']} {item['状态']}")

    print("\n自检完成，没有异常。")
