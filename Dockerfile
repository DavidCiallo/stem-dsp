FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==2.5.4 \
    scipy==1.18.1 \
    soundfile==0.14.0 \
    pyloudnorm==0.2.0

COPY local_dsp_server.py ./
COPY engines/ ./engines/

# 服务默认只监听 127.0.0.1，容器内必须改成 0.0.0.0 才能被端口映射到
ENV DSP_HOST=0.0.0.0 \
    DSP_PORT=8765

EXPOSE 8765

CMD ["python", "local_dsp_server.py"]
