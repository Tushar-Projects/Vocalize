"""
Vocalize — Phase 2: Dataset Processing Pipeline

Batch-processes sign language MP4 videos through the Phase 1 MediaPipe
Holistic extractor.  For each video it saves a NumPy array of shape
(frames, 1662) containing per-frame keypoints.

Expected input layout
─────────────────────
    dataset/
    ├── hello/
    │   ├── 001.mp4
    │   └── 002.mp4
    └── thanks/
        └── 001.mp4

Output layout (mirrors input)
─────────────────────────────
    processed_data/
    ├── hello/
    │   ├── 001.npy      ← shape (F, 1662)
    │   └── 002.npy
    └── thanks/
        └── 001.npy

Usage
─────
    python dataset_processor.py --input_dir dataset --output_dir processed_data
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
from tqdm import tqdm

# ── import Phase 1 helpers ────────────────────
from landmark_extractor import mediapipe_detect, extract_keypoints

import mediapipe as mp

mp_holistic = mp.solutions.holistic

# ──────────────────────────────────────────────
# Video-level processing
# ──────────────────────────────────────────────

def process_video(video_path: str, model: mp_holistic.Holistic) -> np.ndarray:
    """
    Extract keypoints from every frame of an MP4 file.

    Args:
        video_path: Absolute or relative path to the .mp4 file.
        model:      An initialised MediaPipe Holistic instance.

    Returns:
        np.ndarray of shape (num_frames, 1662).

    Raises:
        IOError:  If the video cannot be opened.
        ValueError: If the video contains zero readable frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_sequence: list[np.ndarray] = []

    with tqdm(
        total=total_frames,
        desc="      Frames",
        unit="fr",
        leave=False,
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as frame_bar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            _, results = mediapipe_detect(frame, model)
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)
            frame_bar.update(1)

    cap.release()

    if len(keypoints_sequence) == 0:
        raise ValueError(f"Zero frames extracted from: {video_path}")

    return np.stack(keypoints_sequence, axis=0)


# ──────────────────────────────────────────────
# Dataset-level processing
# ──────────────────────────────────────────────

def discover_videos(input_dir: str) -> list[tuple[str, str, str]]:
    """
    Walk the input directory and return a list of (label, video_stem, video_path)
    tuples for every .mp4 file found under <input_dir>/<label>/.

    Args:
        input_dir: Root directory containing label sub-folders.

    Returns:
        Sorted list of (label, stem, full_path) tuples.
    """
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    entries: list[tuple[str, str, str]] = []

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for filename in sorted(os.listdir(label_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                stem = os.path.splitext(filename)[0]
                full_path = os.path.join(label_dir, filename)
                entries.append((label, stem, full_path))

    return entries


def process_dataset(input_dir: str, output_dir: str) -> dict:
    """
    Process all videos in the dataset, saving per-video .npy files.

    Already-processed videos (output .npy exists) are skipped for
    resumability.

    Args:
        input_dir:  Root of the video dataset.
        output_dir: Root where .npy files will be saved.

    Returns:
        Summary dict with counts for processed, skipped, and failed videos.
    """
    videos = discover_videos(input_dir)
    if not videos:
        print(f"[WARN] No videos found under: {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0, "total": 0}

    # Group by label for reporting
    labels = sorted(set(v[0] for v in videos))
    print(f"\n[INFO] Found {len(videos)} video(s) across {len(labels)} label(s)")
    print(f"[INFO] Labels: {', '.join(labels)}\n")

    stats = {"processed": 0, "skipped": 0, "failed": 0, "total": len(videos)}

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        static_image_mode=False,
    ) as holistic:

        for label, stem, video_path in tqdm(
            videos, desc="Overall", unit="vid", ncols=90
        ):
            # ── output path ──
            label_out_dir = os.path.join(output_dir, label)
            os.makedirs(label_out_dir, exist_ok=True)
            output_path = os.path.join(label_out_dir, f"{stem}.npy")

            # ── skip if already done ──
            if os.path.exists(output_path):
                tqdm.write(f"  [SKIP] {label}/{stem}.npy already exists")
                stats["skipped"] += 1
                continue

            # ── process ──
            try:
                tqdm.write(f"  [PROC] {label}/{stem}.mp4")
                sequence = process_video(video_path, holistic)
                np.save(output_path, sequence)
                tqdm.write(
                    f"  [ OK ] {label}/{stem}.npy  →  shape {sequence.shape}"
                )
                stats["processed"] += 1

            except Exception as exc:
                tqdm.write(f"  [FAIL] {label}/{stem}.mp4  →  {exc}")
                stats["failed"] += 1

    return stats


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vocalize Phase 2 — Batch dataset keypoint extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python dataset_processor.py --input_dir dataset --output_dir processed_data\n"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset",
        help="Root directory containing label sub-folders with MP4 videos (default: dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Root directory where .npy files will be saved (default: processed_data)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Vocalize — Phase 2  Dataset Processor")
    print("=" * 60)
    print(f"  Input:  {os.path.abspath(args.input_dir)}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    t0 = time.time()
    stats = process_dataset(args.input_dir, args.output_dir)
    elapsed = time.time() - t0

    # ── summary ──
    print("\n" + "=" * 60)
    print("  Processing Complete")
    print("=" * 60)
    print(f"  Total videos : {stats['total']}")
    print(f"  Processed    : {stats['processed']}")
    print(f"  Skipped      : {stats['skipped']}")
    print(f"  Failed       : {stats['failed']}")
    print(f"  Elapsed time : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
