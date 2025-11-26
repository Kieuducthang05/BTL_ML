import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd


def collect_annotation_files(annotation_root: str, max_files: int | None = None) -> List[str]:
    pattern = os.path.join(annotation_root, "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if max_files is not None:
        files = files[:max_files]
    return files


def build_frame_level_dataset(
    base_dir: str,
    action: str,
    agent_id: str,
    output_path: str,
    max_files: int | None = None,
    neg_pos_ratio: float | None = None,
) -> None:
    annotation_root = os.path.join(base_dir, "train_annotation")
    norm_root = os.path.join(base_dir, "processed_data_normalized")

    annotation_files = collect_annotation_files(annotation_root, max_files=max_files)
    all_parts: List[pd.DataFrame] = []

    for anno_path in annotation_files:
        lab_id = os.path.basename(os.path.dirname(anno_path))
        video_id = os.path.splitext(os.path.basename(anno_path))[0]

        norm_path = os.path.join(norm_root, lab_id, f"{video_id}_norm.parquet")
        if not os.path.exists(norm_path):
            continue

        anno_df = pd.read_parquet(anno_path)
        required_cols = ["agent_id", "action", "start_frame", "stop_frame"]
        if not all(col in anno_df.columns for col in required_cols):
            raise ValueError(
                f"Missing required columns in annotation file {anno_path}. "
                f"Expected {required_cols}, found {list(anno_df.columns)}."
            )

        try:
            norm_df = pd.read_parquet(norm_path)
        except Exception:
            # File parquet bị lỗi hoặc không đúng định dạng, bỏ qua video này
            continue
        num_frames = len(norm_df)
        labels = np.zeros(num_frames, dtype="int8")

        # Xử lý lọc theo agent_id: hỗ trợ all / mouse1 / số nguyên
        agent_filter_value = None
        agent_id_str = str(agent_id).strip().lower() if agent_id is not None else "all"
        if agent_id_str in ("all", "any", ""):
            subset = anno_df[anno_df["action"] == action]
        else:
            if agent_id_str.startswith("mouse"):
                try:
                    agent_filter_value = int(agent_id_str.replace("mouse", ""))
                except ValueError:
                    agent_filter_value = None
            else:
                try:
                    agent_filter_value = int(agent_id_str)
                except ValueError:
                    agent_filter_value = None

            if agent_filter_value is None:
                subset = anno_df[anno_df["action"] == action]
            else:
                subset = anno_df[(anno_df["action"] == action) & (anno_df["agent_id"] == agent_filter_value)]
        if subset.empty:
            continue

        for _, row in subset.iterrows():
            start = int(row["start_frame"])
            stop = int(row["stop_frame"])
            if stop < 0 or start >= num_frames:
                continue
            start = max(start, 0)
            stop = min(stop, num_frames - 1)
            labels[start : stop + 1] = 1

        part = norm_df.copy()
        if part.index.name is None:
            part.index.name = "video_frame"
        part = part.reset_index()
        part["label"] = labels
        part["video_id"] = video_id
        part["lab_id"] = lab_id

        if neg_pos_ratio is not None and neg_pos_ratio > 0:
            pos_mask = part["label"] == 1
            neg_mask = ~pos_mask
            pos = part[pos_mask]
            neg = part[neg_mask]
            if len(pos) > 0:
                max_neg = int(len(pos) * neg_pos_ratio)
                if len(neg) > max_neg:
                    neg = neg.sample(n=max_neg, random_state=42)
            part = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

        all_parts.append(part)

    if not all_parts:
        raise RuntimeError("No data was collected. Check that normalized files and annotations exist.")

    full_df = pd.concat(all_parts, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--agent-id", type=str, default="mouse1")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--neg-pos-ratio", type=float, default=None)

    args = parser.parse_args()

    build_frame_level_dataset(
        base_dir=args.base_dir,
        action=args.action,
        agent_id=args.agent_id,
        output_path=args.output_path,
        max_files=args.max_files,
        neg_pos_ratio=args.neg_pos_ratio,
    )


if __name__ == "__main__":
    main()
