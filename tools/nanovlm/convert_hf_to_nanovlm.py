#!/usr/bin/env python
"""Convert a Hugging Face dataset to lmms-engine NanoVLM JSON/YAML.

The output JSON keeps lmms-engine's expected OpenAI-style multimodal message
format. Image references are normalized to local relative paths under
``images/`` and are referenced through ``{"type": "image_url"}``.

Example:
    python tools/nanovlm/convert_hf_to_nanovlm.py \
        --repo kcz358/lmms_engine_test \
        --config bagel_example \
        --output-dir /data/lmms_engine_test
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download


TEXT_FALLBACK_KEYS = ("text", "prompt", "question", "query", "instruction", "input")
ANSWER_FALLBACK_KEYS = ("answer", "response", "output", "completion", "label")


def _maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    content = _maybe_json_loads(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [_normalize_content_item(content)]
    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                normalized.append(_normalize_content_item(item))
            else:
                normalized.append({"type": "text", "text": _as_text(item)})
        return normalized
    return [{"type": "text", "text": _as_text(content)}]


def _normalize_content_item(item: dict[str, Any]) -> dict[str, Any]:
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
        return {"type": "text", "text": _as_text(item.get("text", ""))}

    # lmms-engine's convert_open_to_hf expects non-visual content to carry a
    # text field, so unknown content types are made textual instead of passed
    # through as-is.
    return {"type": "text", "text": _as_text(item)}


def _messages_from_fallback_fields(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = ""
    for key in TEXT_FALLBACK_KEYS:
        if row.get(key) not in (None, ""):
            prompt = _as_text(row.get(key))
            break

    answer = ""
    for key in ANSWER_FALLBACK_KEYS:
        if row.get(key) not in (None, ""):
            answer = _as_text(row.get(key))
            break

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def _normalize_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = _maybe_json_loads(row.get("messages"))
    if raw_messages is None:
        return _messages_from_fallback_fields(row)

    if isinstance(raw_messages, dict):
        raw_messages = [raw_messages]
    if not isinstance(raw_messages, list):
        return _messages_from_fallback_fields(row)

    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            messages.append({"role": "user", "content": [{"type": "text", "text": _as_text(message)}]})
            continue
        role = message.get("role", "user")
        messages.append({"role": role, "content": _normalize_content(message.get("content", ""))})
    return messages


def _safe_suffix(path_or_url: str, default: str = ".png") -> str:
    suffix = Path(urlparse(path_or_url).path).suffix
    return suffix or default


class ImageResolver:
    def __init__(self, repo: str, config: str | None, image_dir: Path, timeout: int = 60) -> None:
        self.repo = repo
        self.config = config
        self.image_dir = image_dir
        self.timeout = timeout

    def save_pil_image(self, image: Any, index: int) -> str:
        name = f"{index:06d}.png"
        image.save(self.image_dir / name)
        return name

    def resolve_reference(self, reference: str, index: int, occurrence: int = 0) -> str:
        reference = reference.strip()
        if not reference:
            return ""

        suffix = _safe_suffix(reference)
        name = f"{index:06d}_{occurrence:02d}{suffix}"
        dst = self.image_dir / name
        parsed = urlparse(reference)

        if parsed.scheme in {"http", "https"}:
            if not dst.exists():
                response = requests.get(reference, timeout=self.timeout)
                response.raise_for_status()
                dst.write_bytes(response.content)
            return name

        local_path = Path(reference)
        if local_path.exists():
            if not dst.exists():
                shutil.copy2(local_path, dst)
            return name

        # Treat the value as a path inside the dataset repository.
        try:
            downloaded = hf_hub_download(repo_id=self.repo, repo_type="dataset", filename=reference)
            if not dst.exists():
                shutil.copy2(downloaded, dst)
            return name
        except Exception:
            # Some toy datasets intentionally use a placeholder such as
            # image.png without storing the file. Keep a relative name so the
            # validation script can report it clearly.
            return Path(reference).name


def _find_image_url_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") in {"image", "image_url"}:
                items.append(item)
    return items


def _inject_image_url(messages: list[dict[str, Any]], local_name: str) -> None:
    for message in messages:
        if message.get("role") == "user":
            message.setdefault("content", [])
            message["content"].insert(0, {"type": "image_url", "image_url": {"url": local_name}})
            return
    messages.insert(
        0,
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": local_name}}]},
    )


def convert_dataset(
    repo: str,
    config: str | None,
    split: str,
    output_dir: Path,
    max_samples: int | None,
    drop_text_only: bool,
) -> dict[str, int]:
    if config is None:
        configs = get_dataset_config_names(repo)
        config = "bagel_example" if "bagel_example" in configs else (configs[0] if configs else None)

    dataset = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    image_dir = output_dir / "images"
    json_dir = output_dir / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    resolver = ImageResolver(repo=repo, config=config, image_dir=image_dir)
    rows: list[dict[str, Any]] = []
    stats = {
        "input_rows": 0,
        "output_rows": 0,
        "rows_with_image_column": 0,
        "rows_with_image_url": 0,
        "rows_text_only": 0,
        "rows_dropped_text_only": 0,
    }

    for index, row in enumerate(dataset):
        stats["input_rows"] += 1
        messages = _normalize_messages(row)

        image_column_name = ""
        if row.get("image") is not None:
            image_column_name = resolver.save_pil_image(row["image"], index)
            stats["rows_with_image_column"] += 1

        image_items = _find_image_url_items(messages)
        occurrence = 0
        for item in image_items:
            if item.get("type") == "image":
                item.clear()
                item.update({"type": "image_url", "image_url": {"url": image_column_name}})
                occurrence += 1
                continue

            image_url = item.get("image_url", {})
            reference = image_url.get("url", "") if isinstance(image_url, dict) else _as_text(image_url)
            local_name = resolver.resolve_reference(reference, index, occurrence)
            item["image_url"] = {"url": local_name}
            occurrence += 1

        if image_column_name and not image_items:
            _inject_image_url(messages, image_column_name)

        has_image = bool(_find_image_url_items(messages))
        if has_image:
            stats["rows_with_image_url"] += 1
        else:
            stats["rows_text_only"] += 1
            if drop_text_only:
                stats["rows_dropped_text_only"] += 1
                continue

        rows.append({"id": str(row.get("id", index)), "messages": messages})
        stats["output_rows"] += 1

    json_path = json_dir / "nanovlm_test.json"
    yaml_path = output_dir / "nanovlm_test.yaml"
    stats_path = output_dir / "conversion_stats.json"

    json_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
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
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"repo: {repo}")
    print(f"config: {config}")
    print(f"split: {split}")
    print(f"json: {json_path}")
    print(f"yaml: {yaml_path}")
    print(f"images: {image_dir}")
    print(f"stats: {json.dumps(stats, ensure_ascii=False)}")
    if rows:
        print(f"first sample: {json.dumps(rows[0], ensure_ascii=False)[:2000]}")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="kcz358/lmms_engine_test", help="Hugging Face dataset repo id")
    parser.add_argument("--config", default=None, help="Dataset config/subset. Defaults to bagel_example if present.")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/lmms_engine_test"))
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick smoke tests")
    parser.add_argument("--drop-text-only", action="store_true", help="Drop samples with no image_url after conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        repo=args.repo,
        config=args.config,
        split=args.split,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        drop_text_only=args.drop_text_only,
    )


if __name__ == "__main__":
    main()
