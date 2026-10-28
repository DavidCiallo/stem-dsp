# 六轨分离服务 · 阿里云函数计算部署

## 1. 服务概述

基于 Demucs `htdemucs_6s` 模型的音频分离服务，运行于阿里云函数计算 GPU 实例，通过 HTTP 接口对外提供六轨分离能力。分离结果写入对象存储 OSS，接口返回预签名下载地址。

| 项 | 值 |
|---|---|
| 分离模型 | Demucs `htdemucs_6s` |
| 输出音轨 | vocals、drums、bass、guitar、piano、other |
| 输出格式 | WAV，44.1 kHz，16 bit，立体声 |
| 推理设备 | GPU（NVIDIA A10） |
| 结果存储 | 阿里云 OSS，私有读写 |
| 镜像 | `docker.io/dcolor/stem-dsp:v6` |

## 2. 架构

```
客户端 ──上传音频──▶ 函数计算实例（自定义容器）
                          │
                          ├─ HTTP 服务，监听 8765
                          ├─ Demucs 推理，GPU
                          └─ 写入 ──▶ OSS ──预签名地址──▶ 客户端
```

函数实例启动后执行 Initializer 回调（`POST /initialize`），完成 torch 导入、CUDA 初始化和模型加载。实例就绪后开始接收业务请求。

## 3. 镜像构建

```sh
docker build -f Dockerfile.fc -t dcolor/stem-dsp:v6 .
docker push dcolor/stem-dsp:v6
```

镜像基于 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，内置 Demucs 4.0.1、FFmpeg 与模型权重。构建产物 10.5 GB，压缩后 3.57 GB。

## 4. 函数配置

| 配置项 | 值 |
|---|---|
| 运行环境 | 自定义镜像 |
| 容器镜像 | `docker.io/dcolor/stem-dsp:v6` |
| 启动命令 | `python local_dsp_server.py` |
| 监听端口 | `8765` |
| 执行超时时间 | 900 秒 |
| 实例规格 | GPU 实例（A10） |
| Initializer 类型 | 调用代码 |
| Initializer 超时 | 300 秒 |
| HTTP 触发器认证 | 匿名（业务鉴权由服务自身完成） |

### 环境变量

| 变量 | 说明 |
|---|---|
| `DSP_TOKEN` | 接口鉴权令牌，请求头携带 `Authorization: Bearer <token>` |
| `DSP_OSS_BUCKET` | OSS 存储空间名称 |
| `DSP_OSS_ENDPOINT` | OSS 内网 endpoint，形如 `oss-us-east-1-internal.aliyuncs.com` |
| `DSP_OSS_AK` | AccessKey ID |
| `DSP_OSS_SK` | AccessKey Secret |
| `DSP_OSS_PREFIX` | 对象前缀，默认 `stems` |
| `DSP_OSS_EXPIRES` | 预签名地址有效期，默认 604800 秒（7 天） |
| `DSP_OSS_PUBLIC_ENDPOINT` | 预签名使用的公网 endpoint，默认由内网 endpoint 去掉 `-internal` 得到 |

镜像内已固化 `DSP_HOST`、`DSP_PORT`、`DSP_DEMUCS_PYTHON`、`DSP_DEMUCS_MODEL`、`TORCH_HOME`、`DSP_DEMUCS_DEVICE`。

## 5. 接口

### 5.1 鉴权

除 `/health`、`/api/docs`、`/initialize` 外，所有接口需要请求头：

```
Authorization: Bearer <DSP_TOKEN>
```

### 5.2 端点

| Method | Path | 说明 |
|---|---|---|
| GET | `/health` | 服务状态 |
| GET | `/api/docs` | 接口文档 |
| GET | `/capabilities` | 模块与操作清单 |
| POST | `/upload` | 上传源音频 |
| POST | `/process` | 提交分离任务 |
| POST | `/warmup` | 预热 |
| POST | `/initialize` | 预热，函数计算 Initializer 调用 |
| GET | `/task/{taskId}` | 任务状态 |
| GET | `/tasks` | 最近任务 |

### 5.3 上传

```
POST /upload
Content-Type: audio/mpeg
```

请求体为音频二进制。响应：

```json
{"ok": true, "fileId": "d03d89fb7f3e471591f344d0d94ff426", "bytes": 9582957, "format": "mp3", "duration": 239.511}
```

支持 `audio/mpeg`、`audio/wav`、`audio/flac`、`audio/m4a`。

### 5.4 提交分离

```
POST /process
Content-Type: application/json
```

```json
{"fileId": "d03d89fb7f3e471591f344d0d94ff426", "taskId": "qhc-v6"}
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `fileId` | string | 二选一 | `/upload` 返回的文件标识 |
| `sourceUrl` | string | 二选一 | 源音频下载地址，指向同地域 OSS 对象时用内网域名签名 |
| `taskId` | string | 否 | 任务标识，缺省时自动生成 |
| `stems` | array | 否 | 指定输出音轨，缺省输出全部六轨 |
| `async` | boolean | 否 | 为 true 时立即返回任务状态，结果通过 `/task/{taskId}` 获取 |

响应：

```json
{
  "ok": true,
  "taskId": "qhc-v6",
  "status": "done",
  "operation": "separate",
  "duration": 239.511,
  "seconds": 6.9,
  "deduplicated": false,
  "oss": {"bucket": "ss3m", "prefix": "stems/qhc-v6"},
  "stems": [
    {"name": "vocals", "bytes": 42249838, "url": "https://ss3m.oss-us-east-1.aliyuncs.com/stems%2Fqhc-v6%2Fvocals.wav?OSSAccessKeyId=...&Expires=...&Signature=..."},
    {"name": "drums",  "bytes": 42249838, "url": "https://..."},
    {"name": "bass",   "bytes": 42249838, "url": "https://..."},
    {"name": "guitar", "bytes": 42249838, "url": "https://..."},
    {"name": "piano",  "bytes": 42249838, "url": "https://..."},
    {"name": "other",  "bytes": 42249838, "url": "https://..."}
  ]
}
```

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | boolean | 处理成功 |
| `taskId` | string | 任务标识 |
| `status` | string | 固定 `done` |
| `operation` | string | 固定 `separate` |
| `duration` | number | 源音频时长，秒 |
| `seconds` | number | 分离耗时，秒 |
| `deduplicated` | boolean | 重复 taskId 返回历史结果时为 true |
| `oss.bucket` | string | 存储空间名称 |
| `oss.prefix` | string | 对象前缀 |
| `stems[].name` | string | 音轨名 |
| `stems[].bytes` | number | 文件字节数 |
| `stems[].url` | string | 预签名下载地址 |

### 5.5 预热

```
POST /warmup
```

执行一次 1 秒音频的完整分离，完成 torch 导入、CUDA 初始化与模型加载。响应：

```json
{"ok": true, "model": "htdemucs_6s", "device": "cuda", "seconds": 2.8}
```

`POST /initialize` 行为相同，免鉴权，供函数计算 Initializer 调用。执行成功返回 200，执行失败返回 500，函数计算据此重建实例。

## 6. 结果存储

分离结果写入 OSS，对象路径为 `{DSP_OSS_PREFIX}/{taskId}/{stem}.wav`，默认前缀 `stems`：

```
ss3m/stems/qhc-v6/vocals.wav
ss3m/stems/qhc-v6/drums.wav
ss3m/stems/qhc-v6/bass.wav
ss3m/stems/qhc-v6/guitar.wav
ss3m/stems/qhc-v6/piano.wav
ss3m/stems/qhc-v6/other.wav
```

上传经由内网 endpoint 完成，预签名地址使用公网 endpoint 生成。地址有效期 7 天。

### 6.1 endpoint 选择规则

OSS 为同一存储空间提供内网与公网两个访问域名，选择依据是“谁访问”。

| 访问方 | 用途 | endpoint |
|---|---|---|
| 服务端 | 读取 `sourceUrl` 指向的源音频 | 内网 |
| 服务端 | 写入分离结果 | 内网 |
| 客户端 | 生成交给服务端的 `sourceUrl` | 内网签名 |
| 客户端 | 下载结果音轨 | 公网 |

内网域名格式为 `oss-<region>-internal.aliyuncs.com`，公网为 `oss-<region>.aliyuncs.com`。

签名由客户端本地计算，生成内网域名的预签名地址时，客户端访问该域名并非必要操作，任一持有 AccessKey 的进程均可完成。

同一对象两种域名的读取耗时实测（9.58 MB，服务端与存储空间同区域）：

| endpoint | 耗时 | 流量 |
|---|---|---|
| 内网 | 0.16 s | 内网流量 |
| 公网 | 6.55 s | 公网流量，单独计费 |

客户端侧规则：

- 交给 `/process` 的 `sourceUrl`，用内网域名签名
- 需要自己下载的地址，用公网域名签名（服务端返回的 `stems[].url` 已按此规则生成）

## 7. 性能数据

测试音频：周杰伦《青花瓷》，MP3，9.14 MB，时长 239.5 秒。实例为 NVIDIA A10，模型已加载。

| 环节 | 耗时 |
|---|---|
| 上传源音频 | 2.8 s |
| `/process` 端到端 | 22.3 s |
| 其中分离推理 | 6.9 s |
| 下载单轨（40.3 MB） | 6.9 s |

实例冷启动时 `/health` 返回耗时 13.5 秒，其中容器初始化与应用启动占主要部分。Initializer 完成模型加载后，`/warmup` 耗时 2.8 秒。

## 8. 运行机制

| 机制 | 行为 |
|---|---|
| Initializer | 实例构建阶段由函数计算调用 `POST /initialize`，完成模型加载，实例就绪后接收业务请求 |
| 实例冻结 | 无请求期间实例进入冻结状态，调用阶段恢复 |
| 结果持久化 | 分离结果写入 OSS，实例回收后预签名地址持续有效 |

## 9. 客户端调用流程

```
1. 用户选择音频
2. 客户端上传音频到 OSS，取得内网域名签名的 sourceUrl
3. POST /warmup                        预热实例（异步发出，无需等待返回）
4. POST /process {"sourceUrl": "..."}  提交分离，取得六条预签名地址
5. GET <stems[].url>                   下载各音轨（公网域名）
```

采用 `sourceUrl` 提交时，音频经由客户端到 OSS 的一段传输，服务端从内网读取源文件。采用 `/upload` 提交时，音频直接到达服务端。
