# RunPod 分轨 worker

六轨分离（htdemucs_6s），模型常驻，结果推到对象存储。

## 1. 构建镜像

```sh
docker build -f Dockerfile.runpod -t stem-dsp-runpod .
docker tag stem-dsp-runpod ghcr.io/davidciallo/stem-dsp-runpod:latest
docker push ghcr.io/davidciallo/stem-dsp-runpod:latest
```

私有 registry 需要在 RunPod 账号的 Settings → Container Registry 里配凭据，否则模板拉不下来。

## 2. 建 Endpoint

Templates → New Template：

| 字段 | 值 |
|---|---|
| Container Image | 上面推的镜像地址 |
| Container Disk | 15 GB（镜像 + 模型 + 临时文件） |
| Expose HTTP Ports | 不用填，serverless 走 API 不走端口 |

Endpoint 设置：

| 设置 | 值 | 为什么 |
|---|---|---|
| Idle Timeout | **300 秒** | 值太小的话预热的 worker 在正式请求到达前就被销毁了，白付一次冷启动 |
| Max Workers | 按并发需求 | 一个预热只热一个 worker |
| Execution Timeout | 600 秒 | 5 分钟的歌大约 20–60 秒，留足余量 |
| FlashBoot | 开 | 省掉容器镜像下载那一段 |

## 3. 调用

地址形如 `https://api.runpod.ai/v2/<ENDPOINT_ID>`，鉴权用 RunPod 的 API key。

### 预热

```sh
curl -X POST https://api.runpod.ai/v2/$EP/runsync \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"warmup":true}}'
```

```json
{"ok": true, "warmup": true, "model": "htdemucs_6s", "device": "cuda", "seconds": 0.0}
```

冷 worker 时这个请求要等 20–40 秒（容器启动 + 模型加载），热 worker 时 1 秒内返回。

### 分轨

```sh
curl -X POST https://api.runpod.ai/v2/$EP/runsync \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "sourceUrl": "https://你们的桶/source.wav",
      "uploadUrl": "https://你们的桶/all.zip",
      "taskId": "xxx"
    }
  }'
```

### 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| sourceUrl | string | 二选一 | 源音频地址，worker 自己下载 |
| audioBase64 | string | 二选一 | 源音频 Base64，适合小文件 |
| uploadUrl | string | 是 | 结果 zip 的 PUT 地址 |
| uploadMethod | string | 否 | 上传方法，默认 `PUT` |
| stems | array | 否 | 只出指定分轨，默认全部 |
| quality | string | 否 | `standard`（默认）或 `best`，`best` 是两倍耗时 |
| taskId | string | 否 | 任务标识，原样返回 |
| warmup | boolean | 否 | 为 true 时只预热，不处理音频 |

分轨名固定：`vocals`、`drums`、`bass`、`guitar`、`piano`、`other`。

### 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| ok | boolean | 处理成功 |
| taskId | string | 任务标识 |
| duration | number | 源音频时长，秒 |
| seconds | number | 处理耗时，秒 |
| stems | array | 每个分轨的 `name` 和 `bytes` |
| zip.bytes | number | 结果包字节数 |
| uploadStatus | number | `uploadUrl` 的响应状态码 |

## 4. 两点

**先预热再上传。** 客户端一开始往对象存储传源文件就发 warmup，让 20–40 秒的冷启动和上传并行，别等传完再预热。

**异步调用更省心。** `POST /run` 立刻返回 `{"id":"...","status":"IN_QUEUE"}`，用 `GET /status/<id>` 轮询。`/runsync` 要一直挂着连接。
