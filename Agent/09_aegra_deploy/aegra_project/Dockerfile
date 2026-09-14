# 生产镜像（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）
# aegra up 会用这个文件构建应用容器；逐行中文注释见 03_本地开发与生产部署_jxsd.py
FROM python:3.12-slim

WORKDIR /app

# 国内 PyPI 镜像加速
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 2026
CMD ["aegra", "serve", "--host", "0.0.0.0", "--port", "2026"]
