"""Small OpenAI-compatible server for a merged Arknights LoRA model."""

from __future__ import annotations

import argparse
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "arknights"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)


app = FastAPI(title="Arknights Character LLM")
model = None
tokenizer = None


def load_model(model_path: str, adapter_path: str | None, load_in_4bit: bool) -> None:
    global model, tokenizer
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path and PeftModel is not None:
        base = PeftModel.from_pretrained(base, adapter_path)
    model = base.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    if model is None or tokenizer is None:
        return {"error": "model not loaded"}

    messages = [m.model_dump() for m in request.messages]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature or None,
            top_p=request.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs.shape[1]:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    prompt_tokens = inputs.shape[1]
    completion_tokens = new_tokens.shape[0]

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output/arknights-merged")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    load_model(args.model_path, args.adapter_path, args.load_in_4bit)
    print(f"model loaded, serving on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

