#!/usr/bin/env python
"""Convert a local kcz358/lmms_engine_test download to NanoVLM format.

This script is designed for an already-downloaded Hugging Face dataset repo,
for example:

    hf download kcz358/lmms_engine_test \
      --repo-type dataset \
      --local-dir /data/hf_datasets/lmms_engine_test

It writes:

    <output_dir>/json/nanovlm_test.json
    <output_dir>/images/
    <output_dir>/nanovlm_test.yaml
    <output_dir>/conversion_stats.json

The JSON keeps the OpenAI-style multimodal message format expected by
lmms-engine's qwen3_vl_iterable loader: image inputs are represented as
{"type": "image_url", "image_url": {"url": "<local image filename>"}}.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset


TEXT_FALLBACK_KEYS = ("text", "prompt", "question", "query", "instruction", "input")
ANSWER_FALLBACK_KEYS = ("answer", "response", "output", "completion", "label")


def maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_content_item(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    item_type = item.get("type")

    if item_type == "image":
        return {"type": "image"}
    if item_type == "image_url":
        image_url = item.get("image_url", "")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict):
            return {"type": "image_url", "image_url": {"url": image_url.get("url", "")}}
        return {"type": "image_url", "image_url": {"url": ""}}
    if item_type == "video_url":
        video_url = item.get("video_url", "")
        if isinstance(video_url, str):
            return {"type": "video_url", "video_url": {"url": video_url}}
        if isinstance(video_url, dict):
            return {"type": "video_url", "video_url": video_url}
        return {"type": "video_url", "video_url": {"url": ""}}
    if item_type == "video":
        return {"type": "video_url", "video_url": {"url": item.get("video", item.get("url", ""))}}
    if item_type == "text" or "text" in item:
        return {"type": "text", "text": as_text(item.get("text", ""))}

    # lmms-engine's convert_open_to_hf expects non-visual content to carry a
    # text field, so unknown content types are made textual.
    return {"type": "text", "text": as_text(item)}


def normalize_content(content: Any) -> list[dict[str, Any]]:
    content = maybe_json_loads(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [normalize_content_item(content)]
    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                normalized.append(normalize_content_item(item))
            else:
                normalized.append({"type": "text", "text": as_text(item)})
        return normalized
    return [{"type": "text", "text": as_text(content)}]


def fallback_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = ""
    for key in TEXT_FALLBACK_KEYS:
        if row.get(key) not in (None, ""):
            prompt = as_text(row.get(key))
            break

    answer = ""
    for key in ANSWER_FALLBACK_KEYS:
        if row.get(key) not in (None, ""):
            answer = as_text(row.get(key))
            break

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def normalize_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = maybe_json_loads(row.get("messages"))
    if raw_messages is None:
        return fallback_messages(row)

    if isinstance(raw_messages, dict):
        raw_messages = [raw_messages]
    if not isinstance(raw_messages, list):
        return fallback_messages(row)

    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            messages.append({"role": "user", "content": [{"type": "text", "text": as_text(message)}]})
            continue
        role = message.get("role", "user")
        messages.append({"role": role, "content": normalize_content(message.get("content", ""))})
    return messages


def find_image_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") in {"image", "image_url"}:
                items.append(item)
    return items


def inject_image_url(messages: list[dict[str, Any]], local_name: str) -> None:
    for message in messages:
        if message.get("role") == "user":
            message.setdefault("content", [])
            message["content"].insert(0, {"type": "image_url", "image_url": {"url": local_name}})
            return
    messages.insert(0, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": local_name}}]})


def path_suffix(value: str, default: str = ".png") -> str:
    suffix = Path(urlparse(value).path).suffix
    return suffix or default


class LocalImageResolver:
    def __init__(self, input_dir: Path, image_dir: Path) -> None:
        self.input_dir = input_dir.resolve()
        self.image_dir = image_dir.resolve()
        self._file_index: dict[str, list[Path]] | None = None

    def _build_file_index(self) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        for path in self.input_dir.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path)
        return index

    @property
    def file_index(self) -> dict[str, list[Path]]:
        if self._file_index is None:
            self._file_index = self._build_file_index()
        return self._file_index

    def _find_local_reference(self, reference: str) -> Path | None:
        if not reference:
            return None

        ref_path = Path(reference)
        candidates = []
        if ref_path.is_absolute():
            candidates.append(ref_path)
        candidates.append(self.input_dir / reference)
        candidates.append(self.input_dir / ref_path.name)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        basename_matches = self.file_index.get(ref_path.name, [])
        return basename_matches[0] if basename_matches else None

    def copy_reference(self, reference: str, row_index: int, occurrence: int) -> str:
        reference = reference.strip()
        if not reference:
            return ""

        src = self._find_local_reference(reference)
        suffix = path_suffix(reference)
        dst_name = f"{row_index:06d}_{occurrence:02d}{suffix}"
        dst = self.image_dir / dst_name

        if src is None:
            # Keep a relative path so the validator can report it clearly.
            return Path(reference).name

        if not dst.exists():
            shutil.copy2(src, dst)
        return dst_name

    def save_image_value(self, image: Any, row_index: int) -> str:
        dst_name = f"{row_index:06d}.png"
        dst = self.image_dir / dst_name

        if hasattr(image, "save"):
            image.save(dst)
            return dst_name

        if isinstance(image, dict):
            image_bytes = image.get("bytes")
            image_path = image.get("path")
            if image_bytes:
                dst.write_bytes(image_bytes)
                return dst_name
            if image_path:
                copied = self.copy_reference(str(image_path), row_index, 0)
                return copied or dst_name

        if isinstance(image, (bytes, bytearray)):
            dst.write_bytes(bytes(image))
            return dst_name

        if isinstance(image, str):
            copied = self.copy_reference(image, row_index, 0)
            return copied or dst_name

        raise ValueError(f"Unsupported image column value type: {type(image)}")


def discover_parquet_files(input_dir: Path, config: str | None, split: str) -> list[Path]:
    all_files = sorted(input_dir.rglob("*.parquet"))
    if not all_files:
        return []

    def score(path: Path) -> tuple[int, int, str]:
        text = str(path).lower()
        name = path.name.lower()
        split_score = 0 if split.lower() in name or f"/{split.lower()}" in text.replace("\\", "/") else 1
        config_score = 0
        if config:
            config_score = 0 if config.lower() in text else 1
        elif "bagel_example" in text:
            config_score = -1
        return (config_score, split_score, str(path))

    ranked = sorted(all_files, key=score)
    best_config_score, best_split_score, _ = score(ranked[0])
    selected = [path for path in ranked if score(path)[:2] == (best_config_score, best_split_score)]
    return selected


def load_local_dataset(input_dir: Path, config: str | None, split: str) -> Dataset:
    parquet_files = discover_parquet_files(input_dir, config, split)
    if parquet_files:
        print("Using parquet files:")
        for path in parquet_files:
            print(f"  {path}")
        datasets = [Dataset.from_parquet(str(path)) for path in parquet_files]
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    print("No parquet files found; trying datasets.load_dataset on the local directory.")
    loaded = load_dataset(str(input_dir), config, split=split) if config else load_dataset(str(input_dir), split=split)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(loaded)}")
    return loaded


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    config: str | None,
    split: str,
    max_samples: int | None,
    drop_text_only: bool,
) -> Counter:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    json_dir = output_dir / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_local_dataset(input_dir, config, split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    resolver = LocalImageResolver(input_dir=input_dir, image_dir=image_dir)
    output_rows: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for index in range(len(dataset)):
        row = dict(dataset[index])
        stats["input_rows"] += 1
        messages = normalize_messages(row)

        image_column_name = ""
        if row.get("image") is not None:
            image_column_name = resolver.save_image_value(row["image"], index)
            stats["rows_with_image_column"] += 1

        image_items = find_image_items(messages)
        occurrence = 0
        for item in image_items:
            if item.get("type") == "image":
                item.clear()
                item.update({"type": "image_url", "image_url": {"url": image_column_name}})
                occurrence += 1
                continue

            image_url = item.get("image_url", {})
            reference = image_url.get("url", "") if isinstance(image_url, dict) else as_text(image_url)
            local_name = resolver.copy_reference(reference, index, occurrence)
            item["image_url"] = {"url": local_name}
            occurrence += 1

        if image_column_name and not image_items:
            inject_image_url(messages, image_column_name)

        has_image = bool(find_image_items(messages))
        if has_image:
            stats["rows_with_image_url"] += 1
        else:
            stats["rows_text_only"] += 1
            if drop_text_only:
                stats["rows_dropped_text_only"] += 1
                continue

        output_rows.append({"id": str(row.get("id", index)), "messages": messages})
        stats["output_rows"] += 1

    json_path = json_dir / "nanovlm_test.json"
    yaml_path = output_dir / "nanovlm_test.yaml"
    stats_path = output_dir / "conversion_stats.json"

    json_path.write_text(json.dumps(output_rows, ensure_ascii=False), encoding="utf-8")
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "path": str(json_path),
                        "data_folder": str(image_dir),
                        "data_type": "json",
                    }
                ]
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    stats_path.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input_dir: {input_dir}")
    print(f"output_json: {json_path}")
    print(f"output_yaml: {yaml_path}")
    print(f"output_images: {image_dir}")
    print(f"stats: {dict(stats)}")
    if output_rows:
        print(f"first_sample: {json.dumps(output_rows[0], ensure_ascii=False)[:2000]}")

    return stats


def validate_outputs(output_dir: Path) -> Counter:
    output_dir = output_dir.resolve()
    json_path = output_dir / "json" / "nanovlm_test.json"
    image_dir = output_dir / "images"
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    stats: Counter = Counter()
    missing_images: list[str] = []

    for row_index, row in enumerate(rows):
        stats["rows"] += 1
        row_has_image = False
        row_has_assistant = False
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                row_has_assistant = True
            for item in message.get("content", []):
                item_type = item.get("type", "<missing>")
                stats[f"content_type:{item_type}"] += 1
                if item_type == "image_url":
                    row_has_image = True
                    url = item.get("image_url", {}).get("url", "")
                    if url and not (image_dir / url).exists():
                        missing_images.append(f"row={row_index} url={url}")

        if row_has_image:
            stats["rows_with_image"] += 1
        else:
            stats["rows_text_only"] += 1
        if row_has_assistant:
            stats["rows_with_assistant"] += 1
        else:
            stats["rows_without_assistant"] += 1

    stats["missing_images"] = len(missing_images)
    print(f"validation: {dict(stats)}")
    if missing_images:
        print("missing image examples:")
        for item in missing_images[:50]:
            print(f"  {item}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Local directory created by `hf download kcz358/lmms_engine_test --repo-type dataset --local-dir ...`.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/data/lmms_engine_test"))
    parser.add_argument("--config", default="bagel_example", help="Dataset subset/config hint used to choose parquet files.")
    parser.add_argument("--split", default="train", help="Dataset split hint used to choose parquet files.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for quick tests.")
    parser.add_argument("--drop-text-only", action="store_true", help="Drop samples that have no image_url after conversion.")
    parser.add_argument("--no-validate", action="store_true", help="Skip output validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=args.config,
        split=args.split,
        max_samples=args.max_samples,
        drop_text_only=args.drop_text_only,
    )
    if not args.no_validate:
        validate_outputs(args.output_dir)


if __name__ == "__main__":
    main()
