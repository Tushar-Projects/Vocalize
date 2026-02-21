"""
Vocalize — Data Augmentation for Skeleton Time-Series

Reads processed .npy arrays from processed_data/<word>/ and generates
15 augmented copies per sample using standard skeleton/time-series
augmentation techniques.  Every output keeps the exact (frames, 1662) shape.

Usage
─────
    python augment_data.py
    python augment_data.py --data_dir processed_data --num_augments 15
"""

import os
import sys
import argparse
import glob

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
# Augmentation primitives
# ──────────────────────────────────────────────

def gaussian_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add zero-mean Gaussian noise to all non-zero coordinates."""
    noise = np.random.normal(0, sigma, seq.shape)
    # Preserve zero-padding: only add noise where the original is non-zero
    mask = (seq != 0.0).astype(np.float64)
    return seq + noise * mask


def coordinate_scale(seq: np.ndarray, lo: float = 0.92, hi: float = 1.08) -> np.ndarray:
    """Scale all landmark coordinates by a random factor per axis-group."""
    out = seq.copy()
    # Apply a random global scale to simulate distance variation
    scale = np.random.uniform(lo, hi)
    mask = (out != 0.0).astype(np.float64)
    return out * scale * mask + out * (1 - mask)


def temporal_shift(seq: np.ndarray, max_shift: int = 5) -> np.ndarray:
    """Shift the sequence forward or backward in time (roll with zero pad)."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        shift = 1
    out = np.roll(seq, shift, axis=0)
    if shift > 0:
        out[:shift] = 0.0
    else:
        out[shift:] = 0.0
    return out


def temporal_scale(seq: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """Speed up or slow down the sequence via linear interpolation,
    keeping the original number of frames."""
    n_frames, n_feat = seq.shape
    factor = np.random.uniform(lo, hi)
    new_len = max(2, int(n_frames * factor))

    old_idx = np.linspace(0, n_frames - 1, new_len)
    new_idx = np.linspace(0, n_frames - 1, n_frames)

    out = np.zeros_like(seq)
    for f in range(n_feat):
        out[:, f] = np.interp(new_idx, old_idx, np.interp(old_idx, np.arange(n_frames), seq[:, f]))

    return out


def frame_dropout(seq: np.ndarray, drop_prob: float = 0.10) -> np.ndarray:
    """Randomly zero-out a small percentage of frames."""
    out = seq.copy()
    n_frames = out.shape[0]
    drop_mask = np.random.rand(n_frames) < drop_prob
    out[drop_mask] = 0.0
    return out


def mirror_horizontal(seq: np.ndarray) -> np.ndarray:
    """Mirror x-coordinates to simulate left-right hand swap.
    In the 1662-vector layout:
      pose:  33 × 4 (x,y,z,vis) → indices 0..131,  x at every 4th starting 0
      face:  468 × 3             → indices 132..1535, x at every 3rd starting 132
      lhand: 21 × 3              → indices 1536..1598, x at every 3rd
      rhand: 21 × 3              → indices 1599..1661, x at every 3rd
    Negate x-coordinates and swap left/right hand blocks.
    """
    out = seq.copy()

    # Negate x in pose (every 4th value starting at 0)
    for i in range(0, 132, 4):
        out[:, i] = -out[:, i]

    # Negate x in face (every 3rd value starting at 132)
    for i in range(132, 1536, 3):
        out[:, i] = -out[:, i]

    # Negate x in both hands (every 3rd value)
    for i in range(1536, 1662, 3):
        out[:, i] = -out[:, i]

    # Swap left hand ↔ right hand blocks
    lh = out[:, 1536:1599].copy()
    rh = out[:, 1599:1662].copy()
    out[:, 1536:1599] = rh
    out[:, 1599:1662] = lh

    return out


# ──────────────────────────────────────────────
# Augmentation schedule (15 variants)
# ──────────────────────────────────────────────

def generate_augmentations(seq: np.ndarray, n: int = 15) -> list[np.ndarray]:
    """
    Generate n augmented copies of a single (frames, 1662) array.

    Strategy mix (15 total):
        1-3:   Gaussian noise (small, medium, large σ)
        4-5:   Coordinate scaling (shrink, enlarge)
        6-7:   Temporal shift (left, right)
        8-9:   Temporal scale (speed up, slow down)
       10:     Frame dropout
       11:     Mirror (horizontal flip + hand swap)
       12-15:  Combined (noise + temporal + scale)
    """
    augments: list[np.ndarray] = []

    # 1-3: Noise at increasing levels
    for sigma in [0.003, 0.006, 0.012]:
        augments.append(gaussian_noise(seq, sigma=sigma))

    # 4-5: Coordinate scale
    augments.append(coordinate_scale(seq, lo=0.90, hi=0.95))
    augments.append(coordinate_scale(seq, lo=1.05, hi=1.10))

    # 6-7: Temporal shift
    augments.append(temporal_shift(seq, max_shift=3))
    augments.append(temporal_shift(seq, max_shift=6))

    # 8-9: Temporal scale
    augments.append(temporal_scale(seq, lo=0.80, hi=0.90))
    augments.append(temporal_scale(seq, lo=1.10, hi=1.20))

    # 10: Frame dropout
    augments.append(frame_dropout(seq, drop_prob=0.08))

    # 11: Mirror
    augments.append(mirror_horizontal(seq))

    # 12-15: Combined augmentations
    for _ in range(4):
        aug = seq.copy()
        aug = gaussian_noise(aug, sigma=np.random.uniform(0.003, 0.010))
        aug = coordinate_scale(aug, lo=0.93, hi=1.07)
        if np.random.rand() < 0.5:
            aug = temporal_shift(aug, max_shift=4)
        if np.random.rand() < 0.5:
            aug = temporal_scale(aug, lo=0.85, hi=1.15)
        augments.append(aug)

    return augments[:n]


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def augment_dataset(data_dir: str, num_augments: int = 15) -> dict[str, dict]:
    """
    Walk processed_data/<word>/*.npy and generate `num_augments`
    augmented files alongside each original.

    Returns per-class stats.
    """
    word_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not word_dirs:
        print(f"[ERROR] No subdirectories found in: {data_dir}")
        sys.exit(1)

    stats: dict[str, dict] = {}

    for word in word_dirs:
        word_path = os.path.join(data_dir, word)
        # Only consider originals (skip previously augmented files)
        originals = sorted(
            f for f in glob.glob(os.path.join(word_path, "*.npy"))
            if "_aug" not in os.path.basename(f)
        )

        n_originals = len(originals)
        n_created = 0
        n_skipped = 0

        for npy_path in tqdm(
            originals,
            desc=f"  {word:<12}",
            unit="arr",
            ncols=90,
        ):
            seq = np.load(npy_path)

            # Sanity check shape
            if seq.ndim != 2 or seq.shape[1] != 1662:
                print(f"    [WARN] Unexpected shape {seq.shape} in {npy_path} — skipping")
                continue

            augmented = generate_augmentations(seq, n=num_augments)

            stem = os.path.splitext(os.path.basename(npy_path))[0]
            for idx, aug_arr in enumerate(augmented, start=1):
                aug_path = os.path.join(word_path, f"{stem}_aug{idx}.npy")

                # Skip if already exists (re-runnable)
                if os.path.exists(aug_path):
                    n_skipped += 1
                    continue

                # Final shape assertion
                assert aug_arr.shape == seq.shape, (
                    f"Shape mismatch: {aug_arr.shape} vs {seq.shape}"
                )

                np.save(aug_path, aug_arr)
                n_created += 1

        # Count totals after augmentation
        total_files = len(glob.glob(os.path.join(word_path, "*.npy")))
        stats[word] = {
            "originals": n_originals,
            "created": n_created,
            "skipped": n_skipped,
            "total": total_files,
        }

    return stats


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vocalize — Augment skeleton time-series for training",
    )
    parser.add_argument(
        "--data_dir", type=str, default="processed_data",
        help="Root directory with <word>/*.npy subfolders (default: processed_data)",
    )
    parser.add_argument(
        "--num_augments", type=int, default=15,
        help="Number of augmented copies per original (default: 15)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Vocalize — Data Augmentation")
    print("=" * 60)
    print(f"  Data dir    : {os.path.abspath(args.data_dir)}")
    print(f"  Augments/ea : {args.num_augments}")
    print("=" * 60)

    stats = augment_dataset(args.data_dir, args.num_augments)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Augmentation Complete")
    print("=" * 60)
    print(f"  {'Class':<14} {'Originals':>10} {'Created':>10} {'Total':>10}")
    print("  " + "-" * 46)

    grand_orig = 0
    grand_created = 0
    grand_total = 0

    for word, s in sorted(stats.items()):
        print(f"  {word:<14} {s['originals']:>10} {s['created']:>10} {s['total']:>10}")
        grand_orig += s["originals"]
        grand_created += s["created"]
        grand_total += s["total"]

    print("  " + "-" * 46)
    print(f"  {'TOTAL':<14} {grand_orig:>10} {grand_created:>10} {grand_total:>10}")
    print("=" * 60)
    print(f"\n  Expansion ratio: {grand_total / max(grand_orig, 1):.1f}x")
    print(f"\n  Next step → python train_model.py --data_dir {args.data_dir}\n")


if __name__ == "__main__":
    main()
