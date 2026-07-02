#!/usr/bin/env python
"""Validate a local NanoVLM JSON/YAML dataset.

Example:
    python tools/nanovlm/validate_nanovlm_dataset.py \
        --json /data/lmms_engine_test/json/nanovlm_test.json \
        --image-dir /data/lmms_engine_test/images
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _resolve_from_yaml(path: Path) -> tuple[Path, Path | None]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = config.get("datasets") or []
    if not datasets:
        raise ValueError(f"No datasets found in {path}")
    first = datasets[0]
    json_path = Path(first["path"])
    image_dir = Path(first["data_folder"]) if first.get("data_folder") else None
    return json_path, image_dir


def validate(json_path: Path, image_dir: Path | None) -> Counter:
    rows = _load_json(json_path)
    stats: Counter = Counter()
    missing_images: list[str] = []
    role_counts: Counter = Counter()

    for row_index, row in enumerate(rows):
        stats["rows"] += 1
        if "messages" not in row:
            stats["rows_missing_messages"] += 1
            continue
        messages = row["messages"]
        if not isinstance(messages, list):
            stats["rows_with_non_list_messages"] += 1
            continue

        row_has_image = False
        row_has_assistant = False
        for message in messages:
            if not isinstance(message, dict):
                stats["non_dict_messages"] += 1
                continue
            role = message.get("role", "<missing>")
            role_counts[role] += 1
            if role == "assistant":
                row_has_assistant = True
            content = message.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                stats["non_list_content"] += 1
                continue
            for item in content:
                if not isinstance(item, dict):
                    stats["non_dict_content_items"] += 1
                    continue
                item_type = item.get("type", "<missing>")
                stats[f"content_type:{item_type}"] += 1
                if item_type == "image_url":
                    row_has_image = True
                    url = item.get("image_url", {}).get("url", "")
                    if image_dir is not None and url and not (image_dir / url).exists():
                        missing_images.append(f"row={row_index} url={url}")
                elif item_type not in {"text", "video_url"}:
                    stats["unsupported_content_items"] += 1

        if row_has_image:
            stats["rows_with_image"] += 1
        else:
            stats["rows_text_only"] += 1
        if row_has_assistant:
            stats["rows_with_assistant"] += 1
        else:
            stats["rows_without_assistant"] += 1

    stats["missing_images"] = len(missing_images)

    print(f"json: {json_path}")
    print(f"image_dir: {image_dir}")
    print(f"stats: {dict(stats)}")
    print(f"roles: {dict(role_counts)}")
    if missing_images:
        print("missing image examples:")
        for item in missing_images[:50]:
            print(f"  {item}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=None, help="NanoVLM dataset YAML")
    parser.add_argument("--json", type=Path, default=None, help="NanoVLM JSON file")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing local images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.json
    image_dir = args.image_dir

    if args.yaml is not None:
        yaml_json_path, yaml_image_dir = _resolve_from_yaml(args.yaml)
        json_path = json_path or yaml_json_path
        image_dir = image_dir or yaml_image_dir

    if json_path is None:
        raise SystemExit("Provide --json or --yaml")
    validate(json_path, image_dir)


if __name__ == "__main__":
    main()
