"""LoRA fine-tuning script for the Arknights character chatbot."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_path", default="data/train.jsonl")
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--output_dir", default="output/arknights-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_checkpoint
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model, tokenizer


def prepare_dataset(path: str, tokenizer) -> object:
    dataset = load_dataset("json", data_files=path, split="train")

    def to_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return dataset.map(to_text)


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args)

    train_ds = prepare_dataset(args.data_path, tokenizer)
    eval_ds = prepare_dataset(args.eval_data, tokenizer) if args.eval_data else None

    config_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        bf16_full_eval=args.bf16,
        fp16=args.fp16,
        max_seq_length=args.max_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        gradient_checkpointing=args.grad_checkpoint,
    )
    if args.eval_data:
        config_kwargs["eval_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps
    else:
        config_kwargs["eval_strategy"] = "no"

    training_args = SFTConfig(**config_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    effective_batch = args.batch_size * args.grad_accum
    total_steps = math.ceil(len(train_ds) / effective_batch * args.epochs)
    print(
        f"dataset={len(train_ds)} effective_batch={effective_batch} "
        f"epochs={args.epochs} total_steps≈{total_steps}"
    )
    print("look at the first training log line: `10/x  ...  1.2s/it`;")
    print(f"remaining estimate = ({total_steps} - current_step) * seconds_per_step")

    start_time = time.monotonic()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed_min = (time.monotonic() - start_time) / 60
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training finished, wall time: {elapsed_min:.1f} minutes, trainable params: {trainable}")


if __name__ == "__main__":
    main()
