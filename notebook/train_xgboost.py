import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_classification(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    num_classes = len(np.unique(y_train))
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if num_classes == 2 else "mlogloss"
    model = XGBClassifier(
        n_estimators=750,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        objective=objective,
        eval_metric=eval_metric,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    return model, {"accuracy": acc, "f1_macro": f1}


def train_regression(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    model = XGBRegressor(
        n_estimators=750,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_val, y_pred)
    return model, {"rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="xgb_model.json")
    parser.add_argument("--features-out", type=str, default="xgb_features.npy")
    args = parser.parse_args()

    df = load_data(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in data")

    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])

    # Loại bỏ các cột không phải số (object/string) vì XGBoost không chấp nhận
    non_numeric_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)

    if args.task == "classification":
        model, metrics = train_classification(X, y, args.test_size, args.random_state)
        print("accuracy:", metrics["accuracy"])
        print("f1_macro:", metrics["f1_macro"])
    else:
        model, metrics = train_regression(X, y, args.test_size, args.random_state)
        print("rmse:", metrics["rmse"])
        print("r2:", metrics["r2"])

    model.save_model(args.model_out)
    np.save(args.features_out, X.columns.to_numpy(dtype=str))


if __name__ == "__main__":
    main()
