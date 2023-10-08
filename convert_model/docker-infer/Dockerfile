# Stage 1: Builder/Compiler
FROM python:3.10-slim AS compile-image

# Install dependencies
COPY requirements.txt /
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --user -r /requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime-image
COPY --from=compile-image /root/.local /root/.local
# Environment Setup
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai \
    PATH=/root/.local/bin:$PATH \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
