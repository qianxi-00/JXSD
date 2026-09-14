# -*- coding: utf-8 -*-
"""
MCP ③ 资源（Resource）
================================================================
课案原文：Resources 用于向客户端暴露**只读数据**，如文件内容、数据库查询结果、
配置信息等。客户端可以像访问文件系统一样读取这些资源。

一句话区分三大件：

    # | 装饰器 | 用途 | 客户端调用 | 返回值 |
    # |---|---|---|---|
    # | @mcp.tool | 执行操作（计算、查询、写入） | call_tool(name, args) | 执行结果 |
    # | @mcp.resource | 暴露只读数据 | read_resource(uri) | 数据内容 |
    # | @mcp.prompt | 提供提示词模板 | get_prompt(name, args) | 拼接后的消息 |

资源的核心是 **URI**（不是函数名）。URI 的 scheme 和 path 可以自定义，
建议用有意义的命名体现资源类型。课案给的 URI 设计规范：

    # | 格式 | 说明 | 示例 |
    # |---|---|---|
    # | scheme://path | 固定资源（无参数） | config://app，menu://main |
    # | scheme://path/{param} | 动态资源（单参数） | file://docs/{filename}，users://top/{limit} |
    # | scheme://{a}/{b} | 多级动态（多参数） | api://users/{userId}/posts/{postId} |

本文件按课案顺序完整实现三段，并各加一个「真实版」扩展：

    ① 固定资源      config://app                              —— 课案原样
    ② 动态资源      file://docs/{filename}                    —— 课案原样
                    api://users/{userId}/posts/{postId}       —— 课案「多级动态」规范的落地示例
    ③ 数据库查询类  users://top/{limit}                       —— 真连 PostgreSQL 跑 SQL
                    db://tables                               —— 真查 information_schema
                    db://table/{table}/rows/{limit}           —— 查任意表，带注入防护

数据库连不上时（服务没起 / 口令不对）不会抛异常，而是打印中文提示 + 返回内存里的
演示数据（降级演示路径），保证这节课在任何机器上都能跑完。

课案出处：Agent 课案 → MCP协议 → 资源（了解）

运行方式（本文件自己起服务端，跑完自动关闭）：
    uv run Agent/05_mcp/03_资源_jxsd.py
"""

import asyncio
import json
import re
import socket
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

# 以下为第三方依赖；settings 是**仓库根目录**的统一配置（from config import settings），
# 全仓库只此一份，不要在 _jxsd 代码里另建 conf.py 或改用 os.environ。
import threading
import time

import uvicorn
from fastmcp import FastMCP

from config import settings

# 本文件监听 8010（避开 01/02 的 8000，方便两个文件同时开着对照看）
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8010
MCP_PATH = "/mcp"

mcp = FastMCP("资源演示 🚀")


# ================================================================
# 一、数据库连接：连接串从 settings 拼，绝不硬编码
# ================================================================
# 规范铁律：不硬编码任何密钥、口令、连接串。课案里写的
#     postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph
# 这种字面量必须全部替换掉——口令写进代码就等于把凭据提交进仓库。
#
# 本项目根目录 .env 里已经配好了分字段的数据库信息，两种拼法都行：
#
#   拼法 A（推荐，config.py 已经把拼装逻辑封装成 property，直接取就行）：
#       settings.pg_uri         → LangGraph 用的库（库名 langgraph，本机实测有 6 张表）
#       settings.postgres_url   → 分字段拼出来的 URL：postgresql://<用户>:<口令>@<主机>:<端口>/<库名>
#       settings.postgres_dsn   → psycopg3 的 key=value 形式
#
#   拼法 B（本文用的写法，把拼装过程显式写出来，方便看清每个字段来自哪里）：
#       f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
#       f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
#
# 本文件默认连 settings.pg_uri 指向的库——它是本机**真实有数据**的库
# （langgraph 检查点库，public 下有 checkpoints / checkpoint_blobs 等 6 张表），
# 这样 db://tables 这类资源查出来的就是真表，演示效果最好。
# 想换成 settings.postgres_* 那套库，把下面 PG_URI 换成 settings.postgres_url 即可。
PG_URI = settings.pg_uri or (
    f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
)

# MySQL 版对照（本机当前没起 MySQL，所以只作为「换数据库怎么写」的示范保留）：
#     MYSQL_DSN = dict(
#         host=settings.mysql_host, port=settings.mysql_port,
#         user=settings.mysql_user, password=settings.mysql_password,
#         database=settings.mysql_db, charset="utf8mb4",
#     )
#     然后用 pymysql.connect(**MYSQL_DSN) 换掉下面的 psycopg.connect(PG_URI)。

# 连接失败的降级数据（只在连不上库时使用，避免资源调用直接抛异常）
FALLBACK_USERS = [
    {"name": "张三", "level": 99},
    {"name": "李四", "level": 85},
    {"name": "王五", "level": 72},
]


def query_postgres(sql: str, params: tuple = ()) -> tuple[bool, list | str]:
    """执行一条只读 SQL，返回 (是否成功, 行列表 或 中文错误信息)。

    资源是**只读**语义，所以这里刻意不提供 commit、也不接受多语句；
    psycopg 默认就是「显式事务」，查询完直接关连接不会有副作用。
    连接串里的口令绝不打印，出错信息也只回中文提示。
    """
    try:
        import psycopg
    except ImportError:
        return False, "未安装 psycopg，无法查询 PostgreSQL（安装：uv add psycopg）"

    try:
        # connect_timeout 必给：否则库没起时资源调用会一直吊着，客户端只能干等
        with psycopg.connect(PG_URI, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return True, cur.fetchall()
    except Exception as exc:
        # 只回异常类型 + 一句话，不回完整连接串（里面含口令）
        return False, f"数据库不可用（{type(exc).__name__}）"


# ================================================================
# 二、① 固定资源：URI 里没有任何参数
# ================================================================
# 函数返回什么类型，客户端读到的就是什么类型：
#   - 返回 str  → MCP 包成 TextResourceContents（mimeType=text/plain）
#   - 返回 dict/BaseModel → FastMCP 自动转 JSON 并标 mimeType=application/json
# 课案为了演示「原样返回字符串」，直接返回了一段 JSON 文本；这里保持课案写法。
@mcp.resource("config://app")
def get_app_config() -> str:
    """返回应用配置信息"""
    # 期望输出：['TextResourceContents(... text='\\n    {\\n      "app_name": "计算器服务", ...')]
    return """
    {
      "app_name": "计算器服务",
      "version": "1.0.0",
      "max_precision": 10
    }
    """


# ================================================================
# 三、② 动态资源：URI 模板里带 {参数}，客户端填具体值再读
# ================================================================
# 关键点：@mcp.resource("file://docs/{filename}") 里的 {filename}
# 必须和函数参数名**完全一致**，否则 FastMCP 在注册时就会报错。
# 客户端侧看不到这个资源出现在 list_resources() 里，
# 而是出现在 list_resource_templates() 里（URI 模板列表）。
@mcp.resource("file://docs/{filename}")
def get_file_content(filename: str) -> str:
    """根据文件名返回文档内容"""
    docs = {
        "readme": "# 计算器服务\n\n提供基础数学运算的 MCP 服务。",
        "changelog": "## v1.0.0\n- 支持加减乘除四则运算",
    }
    # 查不到时不抛异常，回一句中文提示——资源读不到内容是很常见的正常情况，
    # 让模型看到「文件不存在」比让连接报错更有用。
    return docs.get(filename, f"文件 '{filename}' 不存在")


# 课案 URI 设计规范里的第三种：scheme://{a}/{b} 多级动态（多参数）。
# 每个 {xxx} 都对应一个函数参数，顺序无所谓，名字必须对上。
@mcp.resource("api://users/{user_id}/posts/{post_id}")
def get_user_post(user_id: str, post_id: str) -> str:
    """多级动态资源示例：读取某个用户的某篇帖子"""
    return json.dumps(
        {"user_id": user_id, "post_id": post_id, "title": f"{user_id} 的第 {post_id} 篇帖子"},
        ensure_ascii=False,
    )


# ================================================================
# 四、③ 数据库查询类资源（本节重点，比现有精简版丰富的一块）
# ================================================================
# 课案原文这个资源是「查内存 list」：
#     users = [{"name": "张三", "level": 99}, ...]
#     return json.dumps(users[:int(limit)])
# 它想讲的其实是「资源可以动态查询后端数据」。这里把它换成**真查 PostgreSQL**，
# 顺带把三个生产上必须处理的点补上：连接失败降级、排序下推到 SQL、表名注入防护。
@mcp.resource("users://top/{limit}")
def get_top_users(limit: int) -> str:
    """查询排名前 N 的用户数据"""
    limit = max(1, min(int(limit), 50))     # 兜底：客户端可能传 0 / 负数 / 超大值

    # 本机 langgraph 库里没有业务用户表，所以用 SQL 的 VALUES 构造一张内联表再 ORDER BY。
    # 这依然是「真连接 + 真 SQL + 真排序」——把它换成你的业务表就是生产代码：
    #     SELECT name, level FROM users ORDER BY level DESC LIMIT %s
    ok, rows = query_postgres(
        "SELECT name, level FROM (VALUES ('张三', 99), ('李四', 85), ('王五', 72)) "
        "AS t(name, level) ORDER BY level DESC LIMIT %s",
        (limit,),
    )

    if not ok:
        # 降级路径：打印中文提示，返回内存演示数据，保证教学流程不断
        print(f"\n⚠️  {rows}")
        print("   → users://top 已降级为内存演示数据。")
        print(f"   请检查 .env 里的 PG_URI / POSTGRES_* 配置，以及 PostgreSQL 是否已启动。")
        return json.dumps(FALLBACK_USERS[:limit], ensure_ascii=False)

    # 期望输出：[{"name": "张三", "level": 99}, {"name": "李四", "level": 85}, ...]
    return json.dumps([{"name": r[0], "level": r[1]} for r in rows], ensure_ascii=False)


@mcp.resource("db://tables")
def list_db_tables() -> str:
    """列出当前数据库里所有业务表（真查 information_schema）"""
    # information_schema 是 PostgreSQL 的系统视图，列出所有 schema / table；
    # 排除 pg_catalog 与 information_schema 自身，剩下的才是业务表。
    ok, rows = query_postgres(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
        "ORDER BY table_schema, table_name"
    # 查询本身不拼业务参数，只读系统视图，所以这段没有任何注入风险。
    )
    if not ok:
        print(f"\n⚠️  {rows} → db://tables 无法读取，返回中文提示。")
        return "数据库不可用：请确认 PostgreSQL 已启动，且 .env 中的 PG_URI 正确。"

    tables = [{"schema": r[0], "table": r[1]} for r in rows]
    # 期望输出类似：[{"schema": "public", "table": "checkpoints"}, ...]
    return json.dumps({"count": len(tables), "tables": tables}, ensure_ascii=False, indent=2)


# 表名不能像参数那样用 %s 占位（表名是标识符不是值，占位符会被当成字符串字面量）。
# 所以只能拼进 SQL —— 拼之前必须**白名单校验**，这是资源类接口最典型的注入点：
#   db://table/users; DROP TABLE users--/rows/10 这种 URI 就是冲这里来的。
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


@mcp.resource("db://table/{table}/rows/{limit}")
def get_table_rows(table: str, limit: int) -> str:
    """查看某张表的前 N 行（表名走严格白名单校验，防 SQL 注入）"""
    if not _IDENT_RE.match(table):
        # 直接拒绝，绝不把可疑字符串送进数据库
        return f"表名非法：{table!r}。只允许字母/数字/下划线，且以字母或下划线开头。"

    limit = max(1, min(int(limit), 20))

    # 只允许读「当前库里真实存在的表」，进一步收窄攻击面：
    # 先查 information_schema 确认存在，再把名字拼进 SQL。
    # 表名不能走 %s 占位（标识符不是值），必须拼串 —— 所以拼之前做「正则 + 存在性」双重校验。
    ok, rows = query_postgres(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s",
        (table,),
    # 存在性校验通过后才拼表名，LIMIT 用整数拼接 —— 两处都不可能带进注入载荷。
    )
    if not ok:
        print(f"\n⚠️  {rows} → db://table 无法读取，返回中文提示。")
        return "数据库不可用：请确认 PostgreSQL 已启动，且 .env 中的 PG_URI 正确。"
    if not rows:
        return f"表 public.{table} 不存在。可先读 db://tables 看有哪些表。"

    # 表名已通过正则 + 存在性双重校验，这里拼接是安全的；LIMIT 用整数拼接同样安全
    ok, data = query_postgres(f'SELECT * FROM public."{table}" LIMIT {limit}')
    if not ok:
        return data

    # rowcount 拿不到，用 cursor.description 才能拿列名——这里简化成按序号列出，
    # 保持示例聚焦在「资源 + 数据库」本身。
    return json.dumps(
        {"table": table, "count": len(data), "rows": [list(r) for r in data]},
        ensure_ascii=False,
        default=str,      # datetime / UUID 等类型转字符串，否则 json 会报错
        indent=2,
    )


# ================================================================
# 五、客户端：课案「客户端读取资源」那段
# ================================================================
# ---------- 六、客户端演示的脚手架：端口探测 → 后台起服务 → 跑客户端 ----------
def _port_in_use(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        return True
    # 端口已被占用时的提示要写清楚「是别人在跑」，否则学员会以为是自己脚本坏了。
    except OSError:
        return False
    finally:
        sock.close()


def start_server_in_thread():
    """后台线程起服务端，供本文件的客户端演示使用；端口被占则直接连已有的。

    返回 None 的语义是「不是我起的」—— 退出时不能去关别人的服务端。
    """
    if _port_in_use(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 已有服务在运行，直接连它。")
        return None

    # mcp.http_app() 把 FastMCP 包成标准 ASGI 应用；用 uvicorn.Server 是为了
    # 拿到 started（等就绪）和 should_exit（干净关闭）两个开关
    app = mcp.http_app(transport="streamable-http", path=MCP_PATH)
    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):          # 轮询等就绪，最多 10 秒；比写死 sleep 可靠
        # 轮询等就绪（最多 10 秒）：用 server.started 判断，而不是写死 sleep。
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ 服务端启动失败：{HTTP_HOST}:{HTTP_PORT} 无法监听。")
    # 起不来就返回 None，调用方据此判断「不关别人的服务」。
    return None


async def main(url: str) -> None:
    from fastmcp import Client

    async with Client(url) as client:
        # --- FastMCP 3.x 把资源分两类 ---
        # list_resources()           ：固定资源（URI 里没有 {参数}）
        # list_resource_templates()  ：动态资源（URI 里含 {参数}），字段名是 uriTemplate
        # 课案客户端那段也是分两次调用，这里保持一致。
        fixed = await client.list_resources()
        templates = await client.list_resource_templates()
        print(f"固定资源: {len(fixed)} 个")
        for r in fixed:
            print(f"  {r.uri}: {r.name} - {r.description}")
        print(f"动态资源: {len(templates)} 个")
        for t in templates:
            # 注意字段名：固定资源是 uri，模板资源是 uriTemplate（课案里也是这么写的）
            print(f"  {t.uriTemplate}: {t.name} - {t.description}")

        # --- 读取固定资源 ---
        # 期望输出：配置信息：[TextResourceContents(uri=AnyUrl('config://app'), ...)]
        # 注意 read_resource 返回的是**列表**（一段 URI 可以对应多份内容），所以下面要取 [0]
        config = await client.read_resource("config://app")
        print(f"\n配置信息：{config}")

        # --- 读取带参数的动态资源（把 {参数} 换成具体值） ---
        # 走的是真数据库查询；连不上库时服务端会降级成内存数据并打印中文提示
        top3 = await client.read_resource("users://top/3")
        print(f"\nTOP3 用户：{top3}")

        # --- 多级动态资源：api://users/{user_id}/posts/{post_id} ---
        # 这一段是为了演示课案 URI 设计规范里的第三种写法（scheme://{a}/{b}）
        post = await client.read_resource("api://users/u-1001/posts/p-07")
        print(f"\n多级动态资源：{post}")

        # --- 数据库元数据资源 ---
        # 真查 information_schema；.text 才是内容正文，直接 print 对象会看到一堆类型信息
        tables = await client.read_resource("db://tables")
        print(f"\n数据库表清单：\n{tables[0].text}")

        # --- 读具体表的前几行（真表；先看 db://tables 里有什么） ---
        # langgraph 的 checkpoints 表一行就有几千字符（存的是整个图状态），
        # 所以这里只取 1 行、并且打印时截断，避免刷屏。
        rows = await client.read_resource("db://table/checkpoints/rows/1")
        preview = rows[0].text
        print(f"\ncheckpoints 前 1 行（截断展示）：\n{preview[:280]}……（共 {len(preview)} 字符）")

        # --- 注入尝试会被挡下来（教学演示） ---
        # "users;DROP" 这种表名会被 _IDENT_RE 白名单直接拒绝，连数据库都不会碰
        evil = await client.read_resource("db://table/users;DROP/rows/1")
        print(f"\n注入尝试的返回：{evil[0].text}")


# ---------- 六、入口：起服务 → 跑客户端演示 → 关掉自己起的服务 ----------
if __name__ == "__main__":
    started = start_server_in_thread()
    url = f"http://{HTTP_HOST}:{HTTP_PORT}{MCP_PATH}"

    try:
        asyncio.run(main(url))
    finally:
        # finally：演示中途报错也要关服务端，否则端口会被占住，下次跑会连到旧实例
        if started:
            server, thread = started
            server.should_exit = True
            thread.join(timeout=10)
            print("\n✅ 本文件启动的服务端已关闭。")
