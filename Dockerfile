FROM ghcr.io/astral-sh/uv:0.8.17 AS uv-bin

# 两个 AI CLI 都放进镜像；登录态在运行时挂载，不进入镜像层。
FROM node:22-slim AS ai-cli
ARG TCLAUDE_VERSION=0.1.5
ARG CLAUDE_CODE_VERSION=2.1.259
RUN npm install --global \
        "@tencent/tclaude@${TCLAUDE_VERSION}" \
        --engine-strict \
        --registry=https://mirrors.tencent.com/npm \
    && npm install --global \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        --registry=https://registry.npmjs.org

FROM python:3.11-slim

# 让 GitHub Container Registry 自动将镜像包关联到本项目。
LABEL org.opencontainers.image.source="https://github.com/wuhangxian/llm-infer-tuner"

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
COPY --from=ai-cli /usr/local/bin/node /usr/local/bin/node
COPY --from=ai-cli /usr/local/lib/node_modules /usr/local/lib/node_modules

# COPY 会解引用上游的相对软链接；在最终镜像中显式恢复入口，确保 npm、
# tclaude 和 claude 的相对依赖都指向已复制的 node_modules。
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s /usr/local/lib/node_modules/@tencent/tclaude/bin/tclaude.js /usr/local/bin/tclaude \
    && ln -s /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude

WORKDIR /app
COPY . /app

# 安装运行时依赖；开发依赖和宿主机的 .venv 不进入镜像。
RUN uv sync --frozen --no-dev \
    && chmod +x /app/*.sh

# Claude CLI 禁止 root 搭配 --dangerously-skip-permissions；使用与远端
# ubuntu 对齐的普通 UID，挂载的 input/output/登录态可直接读写。
RUN groupadd --gid 1000 runner \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash runner \
    && mkdir -p /app/outputs /app/claude-raw-outputs \
    && chown -R runner:runner /app/outputs /app/claude-raw-outputs
ENV HOME=/home/runner
USER runner

# 默认进入项目目录；传入命令时直接执行，例如 run_executor.sh 或 gen_report.sh。
ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["bash"]
