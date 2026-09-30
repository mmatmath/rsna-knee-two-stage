import numpy as np
import pandas as pd


classes = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]

FOLDS = 5


def main():
    df_meta = pd.read_csv("./data/train_meta.csv")
    df_train = pd.read_csv("./data/train.csv")
    df_qwen = pd.read_csv("./data/report_labels_qwen3.csv")
    if not all(frame.StudyInstanceUID.is_unique for frame in (df_meta, df_train, df_qwen)):
        raise ValueError("Duplicate StudyInstanceUID in metadata, train, or Qwen CSV")

    df = df_meta.merge(df_qwen[["StudyInstanceUID"] + classes], on="StudyInstanceUID", how="left")
    gold = df_train[["StudyInstanceUID"] + classes].copy()
    gold = gold.rename(columns={name: name + "_gold" for name in classes})
    df = df.merge(gold, on="StudyInstanceUID", how="left")
    if not df.StudyInstanceUID.is_unique:
        raise ValueError("Merge produced duplicate studies")

    gold_rows = df[[name + "_gold" for name in classes]].notna().any(axis=1)
    for name in classes:
        known_gold = df[name + "_gold"].notna()
        df.loc[known_gold, name] = df.loc[known_gold, name + "_gold"]
        df[name] = df[name].fillna(-1).astype(np.int8)
    df["label_source"] = np.where(gold_rows, "gold", "qwen")
    df = df.drop(columns=[name + "_gold" for name in classes])

    sort_df = df[classes].replace(-1, 0.5).copy()
    sort_df["positive_count"] = (df[classes] == 1).sum(axis=1)
    sort_df["known_count"] = (df[classes] != -1).sum(axis=1)
    df["fold"] = -1
    for source in ("gold", "qwen"):
        subset = sort_df.loc[df.label_source == source]
        order = subset.sort_values(
            ["positive_count", "known_count"] + classes,
            ascending=[False, False] + [False] * len(classes),
            kind="mergesort",
        ).index
        for fold in range(FOLDS):
            df.loc[order[fold::FOLDS], "fold"] = fold
    df["fold"] = df.fold.astype(int)
    df.to_csv("./data/train_5_folds.csv", index=False)
    print(df.groupby(["fold", "label_source"]).size())
    for fold, group in df.groupby("fold"):
        known = (group[classes] != -1).sum()
        positives = (group[classes] == 1).sum()
        print(f"Fold {fold}: known={known.to_dict()}")
        print(f"Fold {fold}: positive_rate={(positives / known.replace(0, np.nan)).round(3).to_dict()}")
    print("Saved: ./data/train_5_folds.csv")


if __name__ == "__main__":
    main()
