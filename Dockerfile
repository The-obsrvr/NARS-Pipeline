FROM nvcr.io/nvidia/pytorch:24.03-py3

# --- OS deps ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl pciutils screen zstd && \
    rm -rf /var/lib/apt/lists/*

ARG OLLAMA_VERSION=v0.23.1
RUN curl -fsSL https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst \
        -o /tmp/ollama.tar.zst && \
    zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar && \
    tar -xf /tmp/ollama.tar -C /usr/local && \
    rm /tmp/ollama.tar.zst /tmp/ollama.tar && \
    ollama --version

# --- Upgrade pip toolchain ---
RUN python -m pip install --no-cache-dir \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0

# --- Install project requirements (includes transformers >= 4.55.0) ---
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# --- User setup ---
ARG uid
ARG gid

# Set derived values with ENV or just use directly
ENV USER_ID=${uid}
ENV USER_GROUP_ID=${gid}
ARG USER=dh
ARG USER_GROUP=dh

RUN addgroup --gid ${USER_GROUP_ID} ${USER_GROUP}
RUN adduser --gecos "" --disabled-password --uid ${USER_ID} --gid ${USER_GROUP_ID} ${USER}

USER ${USER}
COPY . /app

# Expose Ollama port
EXPOSE 11434
