# output/

训练产物目录，不纳入版本控制（见根目录 `.gitignore`）。

按干员名分子目录，约定如下：

```text
output/
├─ 凯尔希-lora/     # LoRA adapter 与训练检查点（约 1.6 GB）
└─ 凯尔希/          # 合并后的完整模型，可被 bot/llm_server.py 直接加载（约 5.8 GB）
```

## 生成方式

```bash
# 1. 训练，产出 LoRA adapter
python train/train_lora.py --output_dir output/凯尔希-lora

# 2. 合并回基座模型
python train/merge_lora.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --adapter output/凯尔希-lora \
  --output_dir output/凯尔希
```

`bot/llm_server.py` 既可以加载合并模型（`--model_path output/凯尔希`），也可以只加载基座 + adapter（`--model_path Qwen/Qwen2.5-3B-Instruct --adapter_path output/凯尔希-lora`）。

## 关于体积

权重单个分片接近 GitHub 的 100 MB 单文件上限，且总计数 GB，因此不随仓库分发。需要共享时建议：

- 上传到 Hugging Face Hub 或网盘，在 README 里放链接；
- 或用 GitHub Releases 挂附件（单文件上限 2 GB）。
