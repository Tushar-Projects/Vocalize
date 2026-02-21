"""
Vocalize — WLASL Dataset Parser

Reads WLASL_v0.3.json and copies videos for a configurable set of target
words from the Kaggle videos/ folder into our project's dataset/<word>/
structure, ready for dataset_processor.py.

Usage
─────
    python wlasl_parser.py --json_path path/to/WLASL_v0.3.json --videos_dir path/to/videos

    Optional:
        --output_dir   Root output directory (default: dataset)
        --words        Comma-separated list of words to override TARGET_WORDS
"""

import os
import sys
import json
import shutil
import argparse

from tqdm import tqdm

# ──────────────────────────────────────────────
# Default target words — extend this list freely
# ──────────────────────────────────────────────
TARGET_WORDS: list[str] = [
    "hello",
    "thanks",
    "iloveyou",
    "please",
    "sorry",
]

# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def load_wlasl_json(json_path: str) -> list[dict]:
    """
    Load and return the WLASL JSON file.

    Args:
        json_path: Path to WLASL_v0.3.json.

    Returns:
        List of gloss entries.
    """
    if not os.path.isfile(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} glosses from {json_path}")
    return data


def find_video_ids(data: list[dict], target_words: list[str]) -> dict[str, list[str]]:
    """
    For each target word, collect all video_id strings from the JSON.

    Args:
        data:         Full WLASL JSON (list of gloss dicts).
        target_words: Words to search for (case-insensitive).

    Returns:
        Dict mapping lowercase word → list of video_id strings.
    """
    # Build a lookup: lowercase gloss → instances
    gloss_lookup: dict[str, list[dict]] = {}
    for entry in data:
        gloss = entry.get("gloss", "").strip().lower()
        gloss_lookup[gloss] = entry.get("instances", [])

    result: dict[str, list[str]] = {}

    for word in target_words:
        key = word.strip().lower()
        if key not in gloss_lookup:
            print(f"  [WARN] '{word}' not found in WLASL JSON — skipping")
            continue

        ids = [
            str(inst["video_id"])
            for inst in gloss_lookup[key]
            if "video_id" in inst
        ]
        result[key] = ids
        print(f"  [FOUND] '{key}' → {len(ids)} video(s)")

    return result


def copy_videos(
    word_to_ids: dict[str, list[str]],
    videos_dir: str,
    output_dir: str,
) -> dict[str, int]:
    """
    Copy the relevant .mp4 files from the Kaggle videos/ directory
    into output_dir/<word>/ structure.

    Args:
        word_to_ids: Mapping from word → list of video_id strings.
        videos_dir:  Path to the Kaggle videos/ folder.
        output_dir:  Root output (default: dataset).

    Returns:
        Dict with counts: copied, missing, already_exists.
    """
    stats = {"copied": 0, "missing": 0, "already_exists": 0}

    total_videos = sum(len(ids) for ids in word_to_ids.values())

    with tqdm(total=total_videos, desc="Copying videos", unit="vid", ncols=90) as pbar:
        for word, video_ids in word_to_ids.items():
            word_dir = os.path.join(output_dir, word)
            os.makedirs(word_dir, exist_ok=True)

            for vid_id in video_ids:
                pbar.set_postfix_str(f"{word}/{vid_id}", refresh=False)

                src = os.path.join(videos_dir, f"{vid_id}.mp4")
                dst = os.path.join(word_dir, f"{vid_id}.mp4")

                if os.path.exists(dst):
                    stats["already_exists"] += 1
                    pbar.update(1)
                    continue

                if not os.path.isfile(src):
                    tqdm.write(f"  [MISS] {word}/{vid_id}.mp4 — source not found, skipping")
                    stats["missing"] += 1
                    pbar.update(1)
                    continue

                shutil.copy2(src, dst)
                stats["copied"] += 1
                pbar.update(1)

    return stats


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vocalize — WLASL dataset parser & file organiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python wlasl_parser.py \\\n"
            "      --json_path  D:/datasets/WLASL/WLASL_v0.3.json \\\n"
            "      --videos_dir D:/datasets/WLASL/videos \\\n"
            "      --output_dir dataset\n"
        ),
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to WLASL_v0.3.json",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Path to the Kaggle videos/ folder containing numbered MP4s",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Root output directory (default: dataset)",
    )
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Comma-separated target words to override the built-in list",
    )
    args = parser.parse_args()

    # ── resolve target words ──
    words = (
        [w.strip() for w in args.words.split(",") if w.strip()]
        if args.words
        else TARGET_WORDS
    )

    print("=" * 60)
    print("  Vocalize — WLASL Parser")
    print("=" * 60)
    print(f"  JSON:       {os.path.abspath(args.json_path)}")
    print(f"  Videos:     {os.path.abspath(args.videos_dir)}")
    print(f"  Output:     {os.path.abspath(args.output_dir)}")
    print(f"  Words ({len(words)}): {', '.join(words)}")
    print("=" * 60)

    # ── step 1: load JSON ──
    data = load_wlasl_json(args.json_path)

    # ── step 2: find video IDs per word ──
    print(f"\n[INFO] Searching for {len(words)} target word(s)...\n")
    word_to_ids = find_video_ids(data, words)

    if not word_to_ids:
        print("[ERROR] No matching words found. Check your target list.")
        sys.exit(1)

    total = sum(len(ids) for ids in word_to_ids.values())
    print(f"\n[INFO] {total} video(s) to copy across {len(word_to_ids)} word(s)\n")

    # ── step 3: copy ──
    stats = copy_videos(word_to_ids, args.videos_dir, args.output_dir)

    # ── summary ──
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"  Copied         : {stats['copied']}")
    print(f"  Already existed : {stats['already_exists']}")
    print(f"  Missing (skip)  : {stats['missing']}")
    print("=" * 60)
    print(f"\n  Next step → run dataset_processor.py --input_dir {args.output_dir}\n")


if __name__ == "__main__":
    main()
