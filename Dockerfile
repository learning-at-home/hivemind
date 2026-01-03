FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL maintainer="Learning@home"
LABEL repository="hivemind"

WORKDIR /home

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  build-essential \
  curl \
  wget \
  git \
  vim \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
  && ln -sf /usr/bin/python3.11 /usr/bin/python \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PYTHON=python3.11

RUN uv pip install --system torch torchvision torchaudio

COPY requirements.txt hivemind/requirements.txt
COPY requirements-dev.txt hivemind/requirements-dev.txt
COPY examples/albert/requirements.txt hivemind/examples/albert/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r hivemind/requirements.txt && \
    uv pip install --system -r hivemind/requirements-dev.txt && \
    uv pip install --system -r hivemind/examples/albert/requirements.txt

COPY . hivemind/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd hivemind && \
    uv pip install --system .[dev]

CMD bash
