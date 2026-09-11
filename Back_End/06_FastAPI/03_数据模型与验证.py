"""FastAPI 数据模型与验证：Pydantic v2 完整用法
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（数据模型与验证）

本节知识点（对应课案的"Field 常用验证参数"与"Pydantic 常用类型"两张表）：
    1. BaseModel：字段声明 + 类型注解 = 一套自动校验规则
    2. Field 常用参数：... 必填 / gt ge lt le / min_length max_length /
       pattern / default / alias / title / description
    3. 注意 Pydantic v1 → v2 的变化：
         · regex     → 改名 pattern（v2 中 regex 已废弃）
         · min_items / max_items → 对列表请改用 min_length / max_length
    4. 常用类型：str/int/float/bool/datetime/date/UUID/EmailStr/HttpUrl/Json
       Optional[X]（= X | None）、List[X]、Dict[K, V]
    5. 自定义校验：@field_validator（单字段）、@model_validator（跨字段）
    6. 模型配置 ConfigDict：populate_by_name（允许用 Python 字段名传参）、
       str_strip_whitespace（自动去首尾空格）、extra="forbid"（拒绝多余字段）
    7. @computed_field 计算字段、model_dump(exclude=...) 的序列化控制
    8. FastAPI 中模型的两个角色：请求体模型（入参）与响应模型（出参）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\03_数据模型与验证.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\03_数据模型与验证.py'
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys
import uuid
from typing import Annotated, Literal, Optional
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8103

app = FastAPI(
    title="03 数据模型与验证",
    description="后端开发基础 · FastAPI 示例：Pydantic 模型与校验",
    version="1.0.0",
)


# ============================================================
# 一、课案的用户模型（v2 写法）
# ============================================================
class Address(BaseModel):
    """嵌套模型：模型里可以再放模型，FastAPI 会递归校验。"""
    province: str = Field(..., min_length=1, max_length=20, description="省/直辖市")
    city: str = Field(..., min_length=1, max_length=20, description="城市")
    detail: str = Field(default="", max_length=100, description="详细地址")


class User(BaseModel):
    """用户模型：把课案的例子完整落到 Pydantic v2。

    字段来源说明：
        - alias="userName"：前端传 camelCase，Python 内部用 snake_case；
          配合 populate_by_name=True 后，两种写法都能接收（测试更友好）。
        - Optional[int] = None：可选字段，必须给默认值，否则就是必填。
    """

    model_config = ConfigDict(
        populate_by_name=True,          # 允许用字段真实名（user_name）传参，测试/ORM 赋值都方便
        str_strip_whitespace=True,      # 自动去掉字符串首尾空格（"  alice  " → "alice"）
        extra="forbid",                 # 出现模型里没声明的字段直接 422，防止悄悄塞字段
        json_schema_extra={
            "examples": [
                {
                    "userName": "alice",
                    "email": "alice@example.com",
                    "age": 28,
                    "birthday": "1996-05-20",
                    "salary": 20000.0,
                    "website": "https://example.com",
                    "hobbies": ["阅读", "跑步"],
                    "phone": "13800138000",
                    "role": "user",
                }
            ]
        },
    )

    # alias：接收前端的 "userName"，Python 内部用 "user_name"
    user_name: str = Field(
        ...,
        alias="userName",
        min_length=2,
        max_length=20,
        pattern=r"^[A-Za-z0-9_]+$",
        title="用户名",
        description="由字母、数字、下划线组成，2~20 位",
    )

    # 必填邮箱（EmailStr 需要 email-validator 包）
    email: EmailStr = Field(..., title="邮箱")

    # 可选整数，18~150 岁
    age: Optional[int] = Field(default=None, ge=18, le=150, title="年龄")

    # 可选日期：字符串 "1996-05-20" 会被自动解析成 date 对象
    birthday: Optional[dt.date] = Field(default=None, title="生日")

    # 必填正数
    salary: float = Field(..., gt=0, title="薪资", description="月薪，单位元")

    # 必填 URL（HttpUrl 会校验协议与主机名格式）
    website: Optional[HttpUrl] = Field(default=None, title="网站")

    # 列表：v2 里用 min_length / max_length（v1 的 min_items / max_items 已废弃）
    hobbies: list[str] = Field(default_factory=list, min_length=0, max_length=5, title="爱好")

    # 手机号正则验证（v2 用 pattern，v1 的 regex 已废弃）
    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$", title="手机号")

    # 字面量类型：只允许这两个值，等价于枚举
    role: Literal["user", "admin"] = Field(default="user", title="角色")

    # 嵌套模型（可选）
    address: Optional[Address] = Field(default=None, title="地址")

    # 客户端不该传的字段：用 exclude=True 让它不出现在文档/序列化里
    internal_note: str = Field(default="", exclude=True, description="内部备注，不对外暴露")

    # ---------- 自定义校验器 ----------
    @field_validator("user_name")
    @classmethod
    def name_not_reserved(cls, value: str) -> str:
        """单字段校验：禁止使用系统保留名。

        field_validator 在类型转换之后运行，所以这里 value 一定是 str。
        raise ValueError 会被 Pydantic 包装成 422 的校验错误，而不是 500。
        """
        reserved = {"admin", "root", "system", "administrator"}
        if value.lower() in reserved:
            raise ValueError(f"用户名 {value!r} 是系统保留名，请换一个")
        return value

    @field_validator("hobbies")
    @classmethod
    def hobbies_unique(cls, value: list[str]) -> list[str]:
        """列表字段校验：去重并去掉空字符串。"""
        cleaned = [h.strip() for h in value if h.strip()]
        if len(cleaned) != len(set(cleaned)):
            # 去重而不是报错：这是"清洗"而不是"拒绝"
            cleaned = list(dict.fromkeys(cleaned))
        return cleaned

    @model_validator(mode="after")
    def check_age_matches_birthday(self) -> "User":
        """跨字段校验：mode="after" 表示所有字段都校验完之后执行。

        适合做"字段之间"的一致性检查，例如：
            - 年龄与生日是否矛盾
            - 开始时间不能晚于结束时间
            - 两次输入的密码是否一致
        """
        if self.age is not None and self.birthday is not None:
            today = dt.date.today()
            real_age = today.year - self.birthday.year - (
                (today.month, today.day) < (self.birthday.month, self.birthday.day)
            )
            if abs(real_age - self.age) > 1:
                raise ValueError(f"年龄({self.age})与生日({self.birthday})明显矛盾")
        return self

    # ---------- 计算字段 ----------
    @computed_field(title="月薪（万元）")           # type: ignore[prop-decorator]
    @property
    def salary_in_wan(self) -> float:
        """计算字段：不属于输入，但会出现在响应与文档里。"""
        return round(self.salary / 10000, 2)


class UserOut(BaseModel):
    """响应模型：只把该给客户端的字段吐出去。

    注意这里没有 salary、没有 internal_note、没有 phone 全量明文 ——
    这就是"响应模型"存在的意义：默认最小暴露。
    """
    user_name: str = Field(..., serialization_alias="userName")
    email: EmailStr
    role: str
    salary_in_wan: float


class Order(BaseModel):
    """订单模型：集中演示 UUID / datetime / list / dict 等常用类型。

    也演示了 default_factory 的用法：
        default=uuid.uuid4 是错的（所有订单会共享同一个 UUID），
        必须写 default_factory=uuid.uuid4，让每次实例化都重新生成。
    """
    order_id: UUID = Field(default_factory=uuid.uuid4, description="订单号，UUID 格式，不传则自动生成")
    created_at: dt.datetime = Field(default_factory=dt.datetime.now, description="创建时间，ISO 8601")
    amount: float = Field(..., gt=0, le=1_000_000, description="金额")
    tags: list[str] = Field(default_factory=list, max_length=3, description="最多 3 个标签")
    extra: dict[str, int] = Field(default_factory=dict, description="任意数字型附加字段")


# ============================================================
# 二、内存"数据库"
# ============================================================
USERS: dict[str, User] = {}


# ============================================================
# 三、接口
# ============================================================
@app.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED,
          summary="创建用户（Pydantic 自动校验）", tags=["用户"])
def create_user(user: User) -> UserOut:
    """收到请求体后，Pydantic 已经把校验做完了；能走到函数体说明数据一定合法。

    这是 FastAPI 最重要的理念之一：
        **校验发生在"进入业务代码之前"**，业务函数里不再需要写一堆 if 判断。
    """
    if user.user_name in USERS:
        # 唯一性这种"需要查库"的约束只能在业务层做 → 抛 HTTPException
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"用户名 {user.user_name} 已存在")
    USERS[user.user_name] = user
    return UserOut(
        user_name=user.user_name,
        email=user.email,
        role=user.role,
        salary_in_wan=user.salary_in_wan,
    )


@app.get("/users/{user_name}", summary="查询用户", tags=["用户"])
def get_user(user_name: str) -> dict:
    """返回原始模型（model_dump），看 exclude=True 字段的效果。"""
    user = USERS.get(user_name)
    if user is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_name} 不存在")
    # exclude=True 的字段不会出现在 dump 结果里
    return user.model_dump(mode="json", by_alias=True)


@app.get("/users", summary="列出所有用户", tags=["用户"])
def list_users() -> dict:
    return {"total": len(USERS), "items": [u.model_dump(by_alias=True) for u in USERS.values()]}


@app.get("/users/{user_name}/schema", summary="查看模型的 JSON Schema（自动生成的校验规则）", tags=["模型"])
def user_schema(user_name: str) -> dict:
    """打印模型自己的 JSON Schema。

    /docs 里的表单、前端校验、代码生成工具，用的都是这份 Schema ——
    定义一次模型，处处受益。路径里的 user_name 只是为了凑出"嵌套路径"的示例，
    实际项目里通常直接用 /schema 或 /openapi.json。
    """
    return User.model_json_schema()


@app.post("/orders", summary="演示各种类型与默认工厂", tags=["模型"])
def create_order(order: Order) -> dict:
    """Pydantic 常用类型总览（都在 Order 模型里用了一遍）：

        UUID           "550e8400-e29b-41d4-a716-446655440000"
        datetime       "2025-01-01T10:00:00"（ISO 8601，带时区也行）
        float          数值范围校验（gt / le）
        list[str]      长度限制用 max_length
        dict[str,int]  键和值都要符合类型
    """
    return {
        "order_id": str(order.order_id),
        "created_at": order.created_at.isoformat(),
        "amount": order.amount,
        "tags": order.tags,
        "extra": order.extra,
        "类型说明": "order_id 是 UUID 对象，created_at 是 datetime 对象，都已自动转换",
    }


# ============================================================
# 自检
# ============================================================
def _age_of(birthday: dt.date) -> int:
    """按"生日是否已过"计算周岁，用于自动生成互相一致的测试数据。"""
    today = dt.date.today()
    return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))


_BIRTHDAY = dt.date(1996, 5, 20)
_AGE = _age_of(_BIRTHDAY)

VALID_USER = {
    "userName": "alice",
    "email": "alice@example.com",
    # 年龄与生日由上面自动算出来，保证两者一致（避免依赖系统日期而校验失败）
    "age": _AGE,
    "birthday": _BIRTHDAY.isoformat(),
    "salary": 20000.0,
    "website": "https://example.com",
    "hobbies": ["阅读", "跑步", "阅读"],     # 故意重复，验证"清洗"逻辑
    "phone": "13800138000",
    "role": "user",
}


def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 数据模型与验证 · 自检")
    print("=" * 72)
    client = TestClient(app)

    def call(desc: str, method: str, path: str, expect: int, **kwargs) -> dict:
        resp = client.request(method, path, **kwargs)
        body = resp.text.replace("\n", " ")
        if len(body) > 200:
            body = body[:200] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {method} {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应 {body}")
        assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"
        return resp.json()

    # ---------- 1. 创建成功 ----------
    data = call("创建用户（合法请求体）", "POST", "/users", 201, json=VALID_USER)
    print("    说明：响应模型 UserOut 只返回 4 个字段，phone/salary/internal_note 都不外泄")
    assert set(data.keys()) == {"userName", "email", "role", "salary_in_wan"}
    assert data["salary_in_wan"] == 2.0

    # ---------- 2. 各类校验失败 ----------
    def bad(desc: str, patch: dict, expect_field: str) -> None:
        payload = {**VALID_USER, "userName": f"user{len(USERS)+len(patch)}", **patch}
        resp = client.post("/users", json=payload)
        detail = resp.json()["detail"][0]
        loc = ".".join(str(x) for x in detail["loc"])
        print(f"\n✓ [{desc}]")
        print(f"    POST /users  状态码 {resp.status_code}（期望 422）")
        print(f"    出错位置 {loc} | 类型 {detail['type']}")
        print(f"    错误说明 {detail['msg']}")
        assert resp.status_code == 422, f"{desc} 期望 422"
        assert expect_field in loc, f"{desc} 应当定位到 {expect_field}，实际 {loc}"

    bad("邮箱格式错误", {"email": "not-an-email"}, "email")
    bad("年龄超出范围", {"age": 200}, "age")
    bad("薪资必须大于 0", {"salary": 0}, "salary")
    bad("手机号格式错误", {"phone": "12345"}, "phone")
    bad("用户名含非法字符", {"userName": "张三!!!"}, "userName")
    bad("保留用户名", {"userName": "admin"}, "userName")
    bad("网站不是合法 URL", {"website": "ftp:/broken"}, "website")
    bad("多余的字段（extra=forbid）", {"unknown_field": 1}, "unknown_field")
    bad("角色取值非法", {"role": "superuser"}, "role")

    # ---------- 3. 跨字段校验 ----------
    print("\n[跨字段校验：年龄与生日矛盾]")
    try:
        User(**{**VALID_USER, "age": max(18, _AGE - 8)})     # 故意把年龄改小 8 岁
        raise AssertionError("应当校验失败")
    except ValidationError as exc:
        msg = exc.errors()[0]["msg"]
        print(f"    ✓ 被拦截：{msg}")

    # ---------- 4. 列表清洗 ----------
    user = User(**VALID_USER)
    print("\n[列表字段清洗]")
    print(f"    传入 hobbies = {VALID_USER['hobbies']}")
    print(f"    清洗后 hobbies = {user.hobbies}（自动去重）")
    assert user.hobbies == ["阅读", "跑步"]

    # ---------- 5. 字符串去空格 ----------
    user2 = User(**{**VALID_USER, "userName": "  bob  "})
    print("\n[字符串自动去首尾空格]")
    print(f"    传入 '  bob  ' → 实际得到 {user2.user_name!r}（str_strip_whitespace=True）")
    assert user2.user_name == "bob"

    # ---------- 6. 别名双写 ----------
    # 注意：不要同时传 alias 和字段名（extra="forbid" 会把多余的那个当错误），
    # 所以先摘掉 "userName"，只用 Python 字段名 user_name 传参。
    data_without_alias = {k: v for k, v in VALID_USER.items() if k != "userName"}
    user3 = User(**{**data_without_alias, "user_name": "carol"})       # populate_by_name=True
    print("\n[别名兼容]")
    print(f"    populate_by_name=True 时，用 Python 字段名 user_name 传参同样有效：{user3.user_name!r}")

    # ---------- 7. 重复用户名 → 409 ----------
    call("重复创建同名用户", "POST", "/users", 409, json=VALID_USER)

    # ---------- 8. 查询与序列化 ----------
    data = call("查询用户（model_dump 结果）", "GET", "/users/alice", 200)
    assert data["userName"] == "alice"
    assert "internal_note" not in data, "exclude=True 的字段不应被序列化"
    print("    ✓ internal_note 字段被 exclude=True 排除，未出现在响应中")

    call("查询不存在的用户", "GET", "/users/nobody", 404)

    # ---------- 9. 其他类型 ----------
    data = call("UUID / datetime / dict 类型演示", "POST", "/orders", 200, json={
        "amount": 199.5, "tags": ["加急", "线上"], "extra": {"coupon": 10},
    })
    assert len(data["order_id"]) == 36, "order_id 应当是标准 UUID 字符串"
    print("    ✓ order_id 默认值由 default_factory 生成，datetime 自动序列化为 ISO 8601")

    # 非法 UUID → 422
    resp = client.post("/orders", json={
        "order_id": "not-a-uuid", "amount": 1, "tags": [], "extra": {},
    })
    print("\n✓ [非法 UUID]")
    print(f"    状态码 {resp.status_code}（期望 422），错误类型 {resp.json()['detail'][0]['type']}")
    assert resp.status_code == 422

    # ---------- 10. JSON Schema ----------
    schema = client.get("/users/alice/schema").json()
    print("\n[自动生成的 JSON Schema 摘要]")
    print(f"    模型标题：{schema.get('title')}")
    print(f"    字段数量：{len(schema.get('properties', {}))}")
    print(f"    必填字段：{schema.get('required')}")
    print(f"    字段清单：{sorted(schema.get('properties', {}).keys())}")
    assert "email" in schema["properties"]

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. Field 是「校验 + 文档 + 默认值」三合一，能用声明就别写 if 判断")
    print("2. v1 → v2 的坑：regex 改名 pattern；列表的 min_items/max_items 改名 min_length/max_length")
    print("3. field_validator 管单个字段，model_validator(mode='after') 管字段之间的关系")
    print("4. ConfigDict：populate_by_name / str_strip_whitespace / extra='forbid' 是生产常用三件套")
    print("5. 请求模型负责「收得严」，响应模型负责「吐得少」，两者不要混用一个类")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 数据模型与验证（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")
    print()

    if "--check" in sys.argv:
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        print()
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
