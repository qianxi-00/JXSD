# -*- coding: utf-8 -*-
"""
MCP ⑩：Docker 部署
================================================================
把 MCP 服务打包成 Docker 镜像部署。

构建与运行：
    docker build -t mcp-life-service .
    docker run -d -p 8000:8000 --name mcp-life mcp-life-service

客户端连接：http://服务器IP:8000/mcp（streamable_http 传输）
"""

# ---- Dockerfile 内容 ----
DOCKERFILE = """
FROM python:3.12-slim

WORKDIR /app

# 安装依赖（uv 更快；这里用 pip 演示，更通用）
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

# 拷贝服务端代码
COPY 05_mcp/01_服务端.py ./

# HTTP 模式运行，监听 8000
EXPOSE 8000
CMD ["uv", "run", "python", "01_服务端.py", "http"]
"""

# 也可用 docker compose：
COMPOSE_YAML = """
services:
  mcp-life:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
"""

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("== Dockerfile ==")
    print(DOCKERFILE)
    print("== docker-compose.yml ==")
    print(COMPOSE_YAML)
