"""Build a character chat dataset from parsed Arknights dialogue records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_target(speaker: str, aliases: set[str]) -> bool:
    return speaker.strip() in aliases


def cap_text(lines: list[str], max_chars: int) -> str:
    text = "\n".join(lines)
    if len(text) > max_chars:
        return "..." + text[-max_chars:]
    return text


def trim_history(history: list[dict], max_messages: int) -> list[dict]:
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def build_samples_for_path(
    turns: list[dict],
    persona: str,
    aliases: set[str],
    max_context_messages: int,
    max_context_chars: int,
    min_text_len: int,
    max_text_len: int,
) -> list[dict]:
    samples: list[dict] = []
    history: list[dict] = []
    pending_user: list[str] = []
    assistant_chunks: list[str] = []

    def emit_and_commit() -> None:
        if not pending_user or not assistant_chunks:
            return
        user_content = cap_text(pending_user, max_context_chars)
        assistant_content = "\n".join(assistant_chunks)
        messages = [
            {"role": "system", "content": persona},
            *trim_history(history, max_context_messages),
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        samples.append({"messages": messages})
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})

    for turn in turns:
        text = turn["text"].strip()
        if len(text) < min_text_len or len(text) > max_text_len:
            continue
        if is_target(turn["speaker"], aliases):
            if pending_user:
                assistant_chunks.append(text)
        else:
            emit_and_commit()
            pending_user.append(f"{turn['speaker']}: {text}")
            assistant_chunks = []

    emit_and_commit()
    return samples


def split_by_path(samples_by_path: dict[str, list[dict]], val_ratio: float):
    train: list[dict] = []
    val: list[dict] = []
    val_paths: list[str] = []
    for path, samples in sorted(samples_by_path.items()):
        digest = hashlib.md5(path.encode("utf-8")).hexdigest()
        if int(digest, 16) % 100 < val_ratio * 100:
            val.extend(samples)
            val_paths.append(path)
        else:
            train.extend(samples)
    return train, val, val_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", default="data/dialogues.jsonl")
    parser.add_argument("--profiles", default="data/profiles.json")
    parser.add_argument("--config", default="config/character.json")
    parser.add_argument("--character", default=None)
    parser.add_argument("--aliases", default=None)
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--val-out", default="data/val.jsonl")
    parser.add_argument("--summary-out", default="data/summary.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.character:
        config["character"] = args.character
        if not args.aliases:
            config["aliases"] = [args.character, *config.get("alias_map", {}).get(args.character, [])]
    if args.aliases:
        config["aliases"] = [a.strip() for a in args.aliases.split(",") if a.strip()]

    character = config["character"]
    aliases = {a.strip() for a in config["aliases"]}
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    records = load_jsonl(Path(args.dialogues))

    profile = profiles.get(character, "")
    if profile:
        profile = profile[: config["persona_max_chars"]]
    persona_parts = [config["system_prompt"]]
    if profile:
        persona_parts.append("角色档案：\n" + profile)
    persona = "\n\n".join(persona_parts)

    by_path: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_path[record["path"]].append(record)

    samples_by_path: dict[str, list[dict]] = {}
    target_line_count = 0
    for path, turns in by_path.items():
        samples = build_samples_for_path(
            turns,
            persona,
            aliases,
            config["max_context_messages"],
            config["max_context_chars"],
            config["min_text_len"],
            config["max_text_len"],
        )
        if samples:
            samples_by_path[path] = samples
        target_line_count += sum(
            1 for turn in turns if is_target(turn["speaker"], aliases)
        )

    train, val, val_paths = split_by_path(samples_by_path, config["val_ratio"])

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.train_out).open("w", encoding="utf-8") as fh:
        for sample in train:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with Path(args.val_out).open("w", encoding="utf-8") as fh:
        for sample in val:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "character": character,
        "total_dialogue_lines": len(records),
        "target_lines": target_line_count,
        "samples": len(train) + len(val),
        "train_samples": len(train),
        "val_samples": len(val),
        "val_paths": val_paths,
    }
    Path(args.summary_out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"character={character} target_lines={target_line_count} "
        f"train={len(train)} val={len(val)}"
    )


if __name__ == "__main__":
    main()
