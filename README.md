# arknights_bot

明日方舟角色扮演对话机器人：从游戏剧情文本出发，构建角色对话数据集，LoRA 微调 Qwen2.5，并接入 QQ 实现自动回复。

整条链路都在这个仓库里跑通：**解析 → 构数据集 → 微调 → 合并权重 → OpenAI 兼容服务 → OneBot 机器人**。

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Base model" src="https://img.shields.io/badge/Base-Qwen2.5--3B--Instruct-6E4AFF">
  <img alt="PEFT" src="https://img.shields.io/badge/Fine--tuning-LoRA%20%2B%204bit-2E7D32">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

---

## 效果与指标

| 项目 | 数值 |
| --- | --- |
| 剧情语料 | 1,577 个文本文件，363,823 条台词 |
| 台词来源 | 剧情正文 277,267 条 + 干员档案 86,556 条 |
| 基座模型 | `Qwen/Qwen2.5-3B-Instruct`，4bit 量化 + LoRA（r=32, alpha=64） |
| 训练环境 | 单卡 24G（AutoDL / RTX 4090） |

已完成两个角色的 LoRA 微调：

| 角色 | 命中台词 | 训练 / 验证样本 | 训练步数 | 最终 train_loss | eval_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 阿米娅 | 6,855 | 2,793 / 193 | 522 | 0.159 | 1.486 → 1.752 |
| 凯尔希 | 5,954 | 2,246 / 61 | 420 | 0.110 | 0.543 → 0.651 |

> eval_loss 在大约 2 个 epoch 后开始回升，属于典型的小验证集过拟合；实际对话效果取 `checkpoint-400`（凯尔希）/ `checkpoint-522`（阿米娅）。

## 链路

```text
剧情文本/*.txt
      │  scripts/parse_story.py
      ▼
dialogues.jsonl (363,823 条结构化台词)  +  profiles.json (干员档案)
      │  scripts/build_dataset.py     ← config/character.json 决定角色人设与别名
      ▼
train.jsonl / val.jsonl  (ChatML 多轮对话)
      │  train/train_lora.py          ← 4bit QLoRA
      ▼
LoRA adapter ──train/merge_lora.py──▶ 合并模型
      │  bot/llm_server.py            ← FastAPI，OpenAI 兼容
      ▼
/v1/chat/completions ──bot/qq_bot.py──▶ OneBot 11 / NapCat ──▶ QQ
```

## 目录结构

```text
arknights_bot/
├─ config/
│  └─ character.json          # 角色人设、别名、上下文与样本过滤参数
├─ scripts/
│  ├─ parse_story.py          # 解析剧本文本 → 结构化台词 + 干员档案
│  └─ build_dataset.py        # 按角色构建 ChatML 训练/验证集
├─ train/
│  ├─ train_lora.py           # QLoRA 微调（transformers + peft + trl）
│  ├─ merge_lora.py           # 合并 adapter 回基座
│  ├─ run_train.sh            # 训练入口（一键参数）
│  └─ requirements.txt
├─ bot/
│  ├─ llm_server.py           # OpenAI 兼容推理服务
│  ├─ qq_bot.py               # OneBot 11 QQ 自动回复
│  ├─ requirements.txt
│  └─ .env.example
├─ remote/
│  ├─ monitor.ps1             # 训练进度终端面板（SSH 读取远端日志）
│  ├─ start-monitor.cmd
│  └─ autostart.sh            # 实例开机自动装依赖并拉起训练
├─ data/                      # 生成产物，不随仓库分发，见 data/README.md
└─ output/                    # LoRA 与合并模型，不随仓库分发，见 output/README.md
```

## 快速开始

### 0. 准备语料

`剧情文本/` 不入库（第三方游戏文本，体积约 10 MB）。请自备解包的剧情文本，按下列结构放置：

```text
剧情文本/总剧情/剧情(utf-8)/
├─ 剧情一览/     # 剧情正文，按「主线/支线/活动」分目录的 .txt
└─ 干员档案/     # 干员档案，按拼音首字母分目录的 .txt
```

解析器识别的行格式很简单——`"说话人": 台词`，以及 `<场景名>` / `【场景名】` 场景标记。不同来源的文本只要满足这个格式即可复用，路径可用 `--root` 覆盖。

### 1. 构建数据集（本地，无需 GPU）

```powershell
python scripts/parse_story.py     # → data/dialogues.jsonl, profiles.json, stats.json
python scripts/build_dataset.py   # → data/train.jsonl, val.jsonl, summary.json
```

这两个脚本只依赖 Python 标准库，不需要装 torch。

`stats.json` 里的 `top_speakers` 会列出每个说话人的台词条数，是选定下一个角色的依据。

### 2. 微调（GPU 服务器）

```bash
pip install -r train/requirements.txt
bash train/run_train.sh
```

默认 `Qwen/Qwen2.5-3B-Instruct` + 4bit LoRA，单卡 24G 约 1 小时跑完。国内机器加镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com bash train/run_train.sh
```

换模型或调超参直接覆盖命令行参数：

```bash
python train/train_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --data_path data/train.jsonl --eval_data data/val.jsonl \
  --lora_r 32 --lora_alpha 64 --epochs 3
```

训练期间可在本地开一个进度面板，实时看步数 / ETA / loss：

```powershell
.\remote\monitor.ps1 -SshTarget root@<your-host> -Port <port> -Key "$HOME\.ssh\id_ed25519"
```

### 3. 合并并启动服务

```bash
python train/merge_lora.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --adapter output/凯尔希-lora \
  --output_dir output/凯尔希

python bot/llm_server.py --model_path output/凯尔希 --host 0.0.0.0 --port 8000
```

也可以跳过合并，直接加载基座 + adapter：

```bash
python bot/llm_server.py \
  --model_path Qwen/Qwen2.5-3B-Instruct \
  --adapter_path output/凯尔希-lora
```

验证服务：

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arknights","messages":[{"role":"user","content":"博士，早上好"}]}'
```

> 这个服务只实现了 `/v1/chat/completions`，没有 `/v1/models`，请直接调对话接口。
> 需要更高吞吐可以换成 `vllm serve output/凯尔希 --served-model-name arknights --port 8000`。

### 4. 接入 QQ

QQ 侧用 [NapCat](https://github.com/NapNeko/NapCatQQ) 或任意 OneBot 11 实现，开一个 WebSocket 服务端口（示例用 `3001`）。

```powershell
Copy-Item bot\.env.example bot\.env
python -u bot\qq_bot.py
```

`bot/.env` 关键配置：

| 变量 | 说明 |
| --- | --- |
| `ONEBOT_WS_URL` | OneBot WebSocket 地址，如 `ws://127.0.0.1:3001` |
| `LLM_API_URL` | 模型服务地址，如 `http://127.0.0.1:8000/v1/chat/completions` |
| `LLM_MODEL` | 请求的模型名，默认 `arknights` |
| `BOT_SYSTEM_PROMPT` | 机器人人设提示词，换角色时改这里 |
| `REPLY_ON_AT_ONLY` | 群聊是否只回复 `@机器人` |
| `REPLY_PROBABILITY` | 非 @ 群消息的回复概率（`REPLY_ON_AT_ONLY=false` 时生效） |
| `ALLOWED_GROUPS` / `ALLOWED_USERS` | 逗号分隔白名单，留空表示不限制 |
| `MAX_HISTORY` / `COOLDOWN_SECONDS` | 上下文轮数 / 同一会话冷却秒数 |

NapCat 的 OneBot WebSocket 服务端配置（`napcat/config/onebot11.json`）：

```json
{
  "network": {
    "websocketServers": [
      {
        "name": "websocket-server",
        "enable": true,
        "host": "127.0.0.1",
        "port": 3001,
        "messagePostFormat": "array",
        "reportSelfMessage": false,
        "token": "",
        "enableForcePushEvent": true,
        "heartInterval": 30000
      }
    ]
  }
}
```

日志出现 `OneBot connected` 即连接成功。建议先私聊测试，再开群聊。

## 换一个角色复刻

整套流程是配置驱动的，换角色**不需要改代码**：

1. 在 `data/stats.json` 的 `top_speakers` 里确认该角色在剧情中实际使用的名字（例如银灰常以本名「恩希欧迪斯」出现）。
2. 改 `config/character.json`：

   | 字段 | 作用 |
   | --- | --- |
   | `character` | 与 `data/profiles.json` 中档案文件名一致的干员名 |
   | `aliases` | 判定「目标角色台词」的说话人名单 |
   | `alias_map` | 其他干员的真名 / 剧情名映射，供 `--character` 快捷切换 |
   | `system_prompt` | 人设提示词，决定角色口吻 |
   | `persona_max_chars` | 档案拼进 system prompt 的字数上限 |
   | `max_context_messages` / `max_context_chars` | 对话历史长度上限 |
   | `min_text_len` / `max_text_len` | 过滤过短 / 过长台词 |

3. 重新生成数据集并查看命中量：

   ```powershell
   python scripts/build_dataset.py --character 银灰
   # 或手动指定别名
   python scripts/build_dataset.py --character 某干员 --aliases 某干员,剧情里的名字
   ```

4. 检查 `data/summary.json` 的 `target_lines` 与 `samples`，太少就补别名。
5. 重新训练、合并，更新 `bot/.env` 的 `BOT_SYSTEM_PROMPT`，重启服务。

如果只想快速试效果，也可以不重训，直接改 `BOT_SYSTEM_PROMPT` 让基座模型扮演；但角色感最强的仍然是按该角色数据重新 LoRA。

## 设计要点

**数据集构建**（`scripts/build_dataset.py`）
以剧情文件为单位切分样本：非目标角色的台词累积为 `user` 侧上下文（保留 `说话人:` 前缀，让模型学会在多人对话里分辨谁在对它说话），目标角色的连续台词合并为一条 `assistant`。训练 / 验证集按文件路径的 MD5 哈希切分而不是随机切分，避免同一场景的相邻轮次同时出现在两侧造成验证集泄漏。

**人设注入**：system prompt = `system_prompt` + 截断后的干员档案，让模型在训练时就绑定「角色口吻 + 角色设定」这一组合。

**显存策略**：4bit NF4 双重量化加载基座，LoRA 只训练注意力与 MLP 的全部投影层（`q/k/v/o/gate/up/down_proj`），配合梯度检查点与 `expandable_segments:True`，3B 模型在 24G 单卡上以 `batch 2 × grad_accum 8` 稳定训练。

**服务层**：`llm_server.py` 把 HuggingFace 模型包成 OpenAI 兼容接口，好处是上游换成本地模型、vLLM 还是远端 API，`qq_bot.py` 一行都不用改。

## 常见问题

- **某角色样本太少** —— 查 `data/stats.json` 的 `top_speakers`，用 `--aliases` 或 `alias_map` 补齐剧情中的称呼。
- **训练 OOM** —— 保持 `--eval_batch_size 1`，并确认设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **`/health` 正常但 `/v1/models` 404** —— 本项目只实现了 `/v1/chat/completions`，属预期行为。
- **模型输出跑偏 / 复读** —— 多为验证集过拟合，取更早的 checkpoint；或提高 `min_text_len` 过滤过短台词。
- **NapCat 报 `The specified module could not be found`** —— 把 QQ 版本包 `resources/app/` 下的 `crypto.dll`、`ssl.dll` 复制到 NapCat 根目录。
- **NapCat 独立 Shell 登录后崩溃（退出码 `3221225477`）** —— 改用注入式 `launcher-user.bat` / `launcher.bat`。

## 说明

- 本仓库**不包含**游戏剧情文本、生成的训练数据与模型权重。语料版权归鹰角网络所有，模型权重体积达数 GB，均不适合入库；请按上面的步骤自行准备与生成。
- 项目仅用于个人学习与技术研究，请勿用于商业用途或公开分发。
- 明日方舟（Arknights）为鹰角网络（Hypergryph）的商标与版权作品，本项目与官方无关。

## License

[MIT](LICENSE)
