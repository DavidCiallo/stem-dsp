FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/models \
    STEM_MODEL=htdemucs_6s

# load_track 解 mp3/flac 要走 ffmpeg，不是只给 wav 用的
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 权重烤进镜像，冷启动不用联网下载
RUN pip install --no-cache-dir "demucs==4.0.1" soundfile runpod \
 && python -c "from demucs.pretrained import get_model; get_model('$STEM_MODEL')" \
 && rm -rf /root/.cache/pip

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
