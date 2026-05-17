#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomly split .mat files from collected_JONS_gamma_mat into
data/sea_surface_simple/train, val, and test folders.

Example usage:

    python scripts/split_jonswap_mat.py --train 800 --val 100 --test 100

If you want reproducible random sampling:

    python scripts/split_jonswap_mat.py --train 800 --val 100 --test 100 --seed 42

If you want to clear existing .mat files in target folders before copying:

    python scripts/split_jonswap_mat.py --train 800 --val 100 --test 100 --clear
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly copy .mat files into train/val/test folders."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="collected_JONS_gamma_mat",
        help="Source folder containing .mat files. Default: collected_JONS_gamma_mat",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="data/sea_surface_rollout",
        help="Target root folder. Default: data/sea_surface_rollout",
    )

    parser.add_argument(
        "--train",
        type=int,
        default=50,
        help="Number of .mat files to copy into train folder.",
    )

    parser.add_argument(
        "--val",
        type=int,
        default=5,
        help="Number of .mat files to copy into val folder.",
    )

    parser.add_argument(
        "--test",
        type=int,
        default=5,
        help="Number of .mat files to copy into test folder.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Default: None.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing .mat files in train/val/test folders before copying.",
    )

    return parser.parse_args()


def clear_mat_files(folder: Path):
    """Remove existing .mat files in a folder."""
    if not folder.exists():
        return

    for file_path in folder.glob("*.mat"):
        file_path.unlink()


def copy_files(files, target_dir: Path):
    """Copy selected files into target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for src_file in files:
        dst_file = target_dir / src_file.name
        shutil.copy2(src_file, dst_file)


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    source_dir = project_root / args.source
    target_root = project_root / args.target

    train_dir = target_root / "train"
    val_dir = target_root / "val"
    test_dir = target_root / "test"

    if args.seed is not None:
        random.seed(args.seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    mat_files = sorted(source_dir.glob("*.mat"))

    if len(mat_files) == 0:
        raise RuntimeError(f"No .mat files found in source folder: {source_dir}")

    total_needed = args.train + args.val + args.test

    if total_needed > len(mat_files):
        raise ValueError(
            f"Not enough .mat files.\n"
            f"Requested: {total_needed}\n"
            f"Available: {len(mat_files)}\n"
            f"Source: {source_dir}"
        )

    sampled_files = random.sample(mat_files, total_needed)

    train_files = sampled_files[: args.train]
    val_files = sampled_files[args.train : args.train + args.val]
    test_files = sampled_files[args.train + args.val :]

    if args.clear:
        print("Clearing existing .mat files in target folders...")
        clear_mat_files(train_dir)
        clear_mat_files(val_dir)
        clear_mat_files(test_dir)

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    print("Done.")
    print(f"Source folder: {source_dir}")
    print(f"Target folder: {target_root}")
    print(f"Train files copied: {len(train_files)} -> {train_dir}")
    print(f"Val files copied:   {len(val_files)} -> {val_dir}")
    print(f"Test files copied:  {len(test_files)} -> {test_dir}")


if __name__ == "__main__":
    main()