"""Parse Arknights raw text files into structured dialogue records."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DIALOGUE_RE = re.compile(r'^"([^"]+)"\s*[:：]\s*(.+)$')
SCENE_RE = re.compile(r'^\s*(?:<[^>]+>|【[^】]+】)\s*$')
UI_HINT_RE = re.compile(
    r"^(初始开放|提升信赖至\d+%以查看更多信息|提升至精英阶段\d以解锁|"
    r"提升至精英阶段2以查看更多信息|解锁新对话|升变至\S+职业以解锁)"
)
PROFILE_STOP_RE = re.compile(r"^(升变档案|晋升记录|干员密录|相关道具|模组)")


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def source_kind(rel: str) -> str:
    if rel.startswith("剧情一览/"):
        return "story"
    if rel.startswith("干员档案/"):
        return "archive"
    return "other"


def chapter_of(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 3:
        return "/".join(parts[1:-1])
    return ""


def parse_file(path: Path, root: Path) -> tuple[list[dict], str | None]:
    text = read_text(path)
    rel = path.relative_to(root).as_posix()
    kind = source_kind(rel)
    records: list[dict] = []
    scene_marker = ""

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if SCENE_RE.match(line):
            scene_marker = line.strip("<>【】")
            continue
        match = DIALOGUE_RE.match(line)
        if not match:
            continue
        speaker = match.group(1).strip()
        content = match.group(2).strip()
        if not speaker or not content:
            continue
        records.append(
            {
                "id": f"{rel}:{line_no}",
                "source": kind,
                "path": rel,
                "chapter": chapter_of(rel),
                "scene": scene_marker or path.stem,
                "turn_index": len(records),
                "speaker": speaker,
                "text": content,
            }
        )

    profile = extract_profile(text) if kind == "archive" else None
    return records, profile


def extract_profile(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("2.相关道具") or "相关道具" in line[:12]:
            break
        if PROFILE_STOP_RE.match(line):
            break
        if DIALOGUE_RE.match(line) or UI_HINT_RE.match(line):
            continue
        if re.match(r"^——\S+$", line):
            continue
        if re.fullmatch(r"[？?■\s]+", line):
            continue
        lines.append(line)

    profile = "\n".join(lines)
    profile = re.sub(r"\n{3,}", "\n\n", profile)
    return profile[:4000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="剧情文本/总剧情/剧情(utf-8)")
    parser.add_argument("--out", default="data/dialogues.jsonl")
    parser.add_argument("--profiles", default="data/profiles.json")
    parser.add_argument("--stats", default="data/stats.json")
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    profiles_path = Path(args.profiles)
    stats_path = Path(args.stats)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speaker_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_files = 0
    total_lines = 0
    profiles: dict[str, str] = {}

    with out_path.open("w", encoding="utf-8") as fh:
        for sub in ("剧情一览", "干员档案"):
            base = root / sub
            if not base.exists():
                continue
            for path in sorted(base.rglob("*.txt")):
                records, profile = parse_file(path, root)
                total_files += 1
                if profile is not None:
                    profiles[path.stem] = profile
                for record in records:
                    speaker_counts[record["speaker"]] += 1
                    source_counts[record["source"]] += 1
                    total_lines += 1
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    profiles_path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = {
        "files": total_files,
        "dialogue_lines": total_lines,
        "by_source": dict(source_counts),
        "top_speakers": speaker_counts.most_common(100),
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"parsed {total_files} files, {total_lines} dialogue lines")
    print(f"top speakers: {', '.join(f'{k}({v})' for k, v in speaker_counts.most_common(10))}")


if __name__ == "__main__":
    main()
