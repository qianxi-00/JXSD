# ============================================================
# Dockerfile —— FastAPI 应用的镜像构建文件（多阶段构建 + uv）
# ============================================================
# 对应课案章节：后端开发基础 → 容器部署 → Docker
#
# 构建与运行（在 Back_End 目录下执行，因为 COPY 用的是相对本文件的路径）：
#   docker build -f 09_容器部署/Dockerfile -t backend-demo:v1.0 .
#   docker run --rm -p 8000:8000 backend-demo:v1.0
#   浏览器打开 http://127.0.0.1:8000/docs
#
# 本项目镜像里跑的是 10_综合实战/book_api 这个完整项目（也可换成任意 FastAPI 应用）。
# ============================================================

# ---------- 第 1 阶段：构建依赖（builder） ----------
# 用 uv 官方镜像：内置 uv 与 Python 3.12，体积比 python:3.12 小、装包快 10 倍以上。
# 固定 bookworm-slim 而不是 latest：避免基础镜像悄悄变化导致构建不可复现。
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV_INDEX_URL：指定国内镜像加速（不写则用官方 PyPI）
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# UV_COMPILE_BYTECODE：安装时预编译 .pyc，容器启动更快
ENV UV_COMPILE_BYTECODE=1
# UV_LINK_MODE=copy：不用硬链接（跨文件系统时硬链接会失败），更稳
ENV UV_LINK_MODE=copy
# UV_PYTHON_DOWNLOADS=never：禁止在线下载 Python，必须用镜像自带的
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 先只复制"依赖声明文件"，再装依赖 —— 这样只要依赖没变，改代码就不会让这层缓存失效
COPY pyproject.toml uv.lock ./

# --no-install-project：只装第三方依赖，不装项目本身（项目本身在第 2 阶段复制）
# --no-dev：不装开发依赖（pytest / ruff 等），镜像更小
RUN uv sync --frozen --no-install-project --no-dev


# ---------- 第 2 阶段：运行环境（runtime） ----------
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE：不生成 .pyc（容器里不需要，避免污染挂载目录）
ENV PYTHONDONTWRITEBYTECODE=1
# PYTHONUNBUFFERED：日志不缓冲，docker logs 能实时看到输出
ENV PYTHONUNBUFFERED=1
# 通过 .pth 机制把 /app 加入 sys.path，这样代码里 `from config import settings` 能生效
ENV PYTHONPATH=/app

# 时区：容器默认 UTC，设置成上海时区，日志时间才符合直觉
ENV TZ=Asia/Shanghai

WORKDIR /app

# 从构建阶段把"装好依赖的虚拟环境"复制过来（这是多阶段构建的核心收益：
# 编译工具链留在 builder 里，最终镜像只带运行时需要的文件）
COPY --from=builder /app/.venv /app/.venv
# 让容器里的 python/pip 直接用这个虚拟环境
ENV PATH="/app/.venv/bin:$PATH"

# 复制应用代码（.dockerignore 会排除 .venv、data、__pycache__ 等）
COPY config.py ./
COPY Back_End/10_综合实战/book_api ./Back_End/10_综合实战/book_api
COPY Back_End/data/.gitkeep ./Back_End/data/.gitkeep

# 创建非 root 用户：容器里用 root 跑服务是安全反模式
# （一旦容器被攻破，攻击者直接拿到 root；K8S 的 PodSecurityPolicy 也会拒绝 root 容器）
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/Back_End/data \
    && chown -R appuser:appuser /app
USER appuser

# 声明容器监听端口（只是文档作用，真正暴露端口靠 docker run -p）
EXPOSE 8000

# 健康检查：Docker 会周期性执行它，失败时容器状态变为 unhealthy
# --interval=30s 每 30 秒检查一次；--start-period=10s 给应用 10 秒启动时间
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

# 启动命令：exec 形式（不经过 shell），这样 SIGTERM 能直接送达 Python 进程，
# 容器停止时 uvicorn 能优雅关闭（等待正在处理的请求结束）。
# 必须监听 0.0.0.0：监听 127.0.0.1 的话容器外永远访问不到。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/Back_End/10_综合实战/book_api"]
