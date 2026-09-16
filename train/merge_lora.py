"""Merge a trained LoRA adapter back into the base model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="output/arknights-lora")
    parser.add_argument("--output_dir", default="output/arknights-merged")
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

