# 环境同步

本项目使用 `uv` 管理 Python 虚拟环境和依赖。

## 前置条件

- 安装 `uv`
- 安装 Python 3.12

## 拉取后同步环境

```powershell
git clone https://github.com/qianxi-00/JXSD.git
cd JXSD
uv sync --locked
```

`uv sync --locked` 会根据 `uv.lock` 创建或更新项目虚拟环境，并安装锁定版本的依赖。

## 激活虚拟环境

PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD：

```bat
.venv\Scripts\activate.bat
```

macOS/Linux：

```bash
source .venv/bin/activate
```

也可以不激活环境，直接使用 `uv run` 执行命令，例如：

```powershell
uv run python --version
```

## 更新依赖

修改 `pyproject.toml` 后执行：

```powershell
uv lock
uv sync --locked
```
