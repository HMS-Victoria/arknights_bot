# data/

这个目录存放**生成产物**，不纳入版本控制（见根目录 `.gitignore`）。克隆仓库后目录为空，按下面顺序重新生成即可。

```powershell
python scripts/parse_story.py     # 需要先准备 剧情文本/ 原始语料
python scripts/build_dataset.py   # 默认读取 config/character.json
```

| 文件 | 生成者 | 内容 |
| --- | --- | --- |
| `dialogues.jsonl` | `parse_story.py` | 全部结构化台词，每行一条 JSON 记录 |
| `profiles.json` | `parse_story.py` | 干员档案文本，`{干员名: 档案文本}` |
| `stats.json` | `parse_story.py` | 文件数、台词数、`top_speakers`（各说话人台词条数） |
| `train.jsonl` | `build_dataset.py` | 训练集，每行 `{"messages": [...]}` |
| `val.jsonl` | `build_dataset.py` | 验证集，按剧情文件哈希切分，避免同文件泄漏 |
| `summary.json` | `build_dataset.py` | 目标角色、命中台词数、样本数、验证集文件列表 |

### `dialogues.jsonl` 单条记录字段

```json
{
  "id": "剧情一览/主线/00.黑暗时代·上/01.序章·上.txt:12",
  "source": "story",
  "path": "剧情一览/主线/00.黑暗时代·上/01.序章·上.txt",
  "chapter": "主线/00.黑暗时代·上",
  "scene": "序章",
  "turn_index": 3,
  "speaker": "阿米娅",
  "text": "博士，请下命令吧。"
}
```

### 训练样本格式

`train.jsonl` / `val.jsonl` 每行是一个完整的多轮对话，已经套好角色人设：

```json
{
  "messages": [
    {"role": "system", "content": "<system_prompt>\n\n角色档案：\n<profile>"},
    {"role": "user", "content": "凯尔希: 我们该走了。"},
    {"role": "assistant", "content": "……我知道。"}
  ]
}
```

用户侧内容保留 `说话人: 台词` 前缀，让模型学会在多人对话里分辨谁在对自己说话。
