import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def load_model(model_path: str) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def build_segments_for_video(
    df_video: pd.DataFrame,
    prob_col: str,
    pred_col: str,
    min_len: int,
) -> List[dict]:
    df_video = df_video.sort_values("video_frame")
    frames = df_video["video_frame"].to_numpy()
    preds = df_video[pred_col].to_numpy()
    probs = df_video[prob_col].to_numpy()

    segments: List[dict] = []
    n = len(frames)
    if n == 0:
        return segments

    in_seg = False
    start_idx = 0

    for i in range(n):
        if preds[i] == 1 and not in_seg:
            in_seg = True
            start_idx = i
        elif preds[i] == 0 and in_seg:
            end_idx = i - 1
            length = end_idx - start_idx + 1
            if length >= min_len:
                seg_prob = float(probs[start_idx : end_idx + 1].mean())
                segments.append(
                    {
                        "start_frame": int(frames[start_idx]),
                        "stop_frame": int(frames[end_idx]),
                        "score": seg_prob,
                    }
                )
            in_seg = False

    if in_seg:
        end_idx = n - 1
        length = end_idx - start_idx + 1
        if length >= min_len:
            seg_prob = float(probs[start_idx : end_idx + 1].mean())
            segments.append(
                {
                    "start_frame": int(frames[start_idx]),
                    "stop_frame": int(frames[end_idx]),
                    "score": seg_prob,
                }
            )

    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--agent-id", type=str, default="mouse1")
    parser.add_argument("--target-id", type=str, default="mouse2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--output-csv", type=str, required=True)

    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)

    feature_names = np.load(args.features_path)
    feature_names = feature_names.tolist()

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing[:10]} ...")

    X = df[feature_names]

    model = load_model(args.model_path)

    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        pos_prob = proba[:, 0]
    else:
        pos_prob = proba[:, 1]

    df["prob"] = pos_prob.astype("float32")
    df["pred"] = (df["prob"] >= args.threshold).astype("int8")

    if "video_id" not in df.columns:
        raise ValueError("Column 'video_id' is required in data for segmentation.")
    if "video_frame" not in df.columns:
        raise ValueError("Column 'video_frame' is required in data for segmentation.")

    all_rows: List[dict] = []
    row_id = 0

    for video_id, df_video in df.groupby("video_id"):
        segments = build_segments_for_video(df_video, prob_col="prob", pred_col="pred", min_len=args.min_len)
        for seg in segments:
            all_rows.append(
                {
                    "row_id": row_id,
                    "video_id": video_id,
                    "agent_id": args.agent_id,
                    "target_id": args.target_id,
                    "action": args.action,
                    "start_frame": seg["start_frame"],
                    "stop_frame": seg["stop_frame"],
                    "score": seg["score"],
                }
            )
            row_id += 1

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
