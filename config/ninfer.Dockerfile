ARG CUDA_IMAGE
FROM ${CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake make gcc-13 g++-13 pkg-config patchelf \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
