#!/usr/bin/env python
"""Inspect a Hugging Face dataset before converting it for NanoVLM.

Example:
    python tools/nanovlm/inspect_hf_dataset.py \
        --repo kcz358/lmms_engine_test \
        --rows 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


def _maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _iter_content_items(messages: Any):
    messages = _maybe_json_loads(messages)
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            yield {"type": "text", "text": content}
        elif isinstance(content, dict):
            yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    yield {"type": "text", "text": item}
                elif isinstance(item, dict):
                    yield item


def _summarize_rows(dataset, rows: int) -> Counter:
    counts: Counter = Counter()
    for row in dataset.select(range(min(rows, len(dataset)))):
        counts["rows_seen"] += 1
        if "messages" in row:
            counts["rows_with_messages"] += 1
            has_image = False
            has_video = False
            for item in _iter_content_items(row["messages"]):
                item_type = item.get("type", "<missing>")
                counts[f"content_type:{item_type}"] += 1
                if item_type in {"image_url", "image"}:
                    has_image = True
                if item_type in {"video_url", "video"}:
                    has_video = True
            if has_image:
                counts["rows_with_image_in_messages"] += 1
            if has_video:
                counts["rows_with_video_in_messages"] += 1
        if row.get("image") is not None:
            counts["rows_with_image_column"] += 1
    return counts


def inspect_dataset(repo: str, configs: list[str] | None, split: str, rows: int) -> None:
    all_configs = get_dataset_config_names(repo)
    selected_configs = configs or all_configs

    print(f"repo: {repo}")
    print(f"configs: {all_configs}")
    print(f"selected configs: {selected_configs}")
    print()

    for config in selected_configs:
        print("=" * 80)
        print(f"config: {config}")
        try:
            splits = get_dataset_split_names(repo, config)
        except TypeError:
            splits = get_dataset_split_names(repo)
        print(f"splits: {splits}")

        if split not in splits:
            print(f"skip: split {split!r} not found")
            continue

        dataset = load_dataset(repo, config, split=split)
        print(dataset)
        print(f"features: {dataset.features}")

        counts = _summarize_rows(dataset, rows=min(rows, len(dataset)))
        print(f"summary over first {min(rows, len(dataset))} rows: {dict(counts)}")

        preview_count = min(rows, len(dataset))
        for index in range(preview_count):
            row = dataset[index]
            print("-" * 80)
            print(f"row {index} keys: {list(row.keys())}")
            for key, value in row.items():
                value = _maybe_json_loads(value)
                text = repr(value)
                if len(text) > 2000:
                    text = text[:2000] + "...<truncated>"
                print(f"{key}: {type(value).__name__}: {text}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="kcz358/lmms_engine_test", help="Hugging Face dataset repo id")
    parser.add_argument("--config", action="append", help="Dataset config/subset to inspect. Can be repeated.")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--rows", type=int, default=3, help="Rows to preview per config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_dataset(args.repo, args.config, args.split, args.rows)


if __name__ == "__main__":
    main()
