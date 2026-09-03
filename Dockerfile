FROM ghcr.io/astral-sh/uv:0.8.17 AS uv-bin

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        jq \
        openssh-client \
        sshpass \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /bin/

WORKDIR /app
COPY . /app

# 安装运行时依赖；开发依赖和宿主机的 .venv 不进入镜像。
RUN uv sync --frozen --no-dev \
    && chmod +x /app/*.sh

# 默认进入项目目录；传入命令时直接执行，例如 run_executor.sh 或 gen_report.sh。
ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["bash"]
