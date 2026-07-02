from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
AGE_BINS = [(8, 13), (13, 18), (18, 23), (23, 28), (28, 33), (33, 38), (38, 43), (43, 48), (48, 53), (53, 58), (58, 63)]
AGE_GROUPS = ["<=15", "16-25", "26-50", ">50"]
CNN_MODEL_NAMES = ["ResNet50", "DenseNet121", "InceptionV3", "InceptionResNetV2", "EfficientNetV2M"]
CNN_BEST3_MODELS = ["ResNet50", "InceptionResNetV2", "DenseNet121"]
VIT_MODEL_NAMES = ["SwinV2_Tiny", "MobileViT_XXS", "ConvNeXtV2_Tiny", "TinyViT_11M"]
DIT_FILE_STEMS = {
    "DiT-Base": "microsoft__dit-base",
    "DiT-Large": "microsoft__dit-large",
    "DiT-Base (RVL-CDIP)": "microsoft__dit-base-finetuned-rvlcdip",
    "DiT-Large (RVL-CDIP)": "microsoft__dit-large-finetuned-rvlcdip",
}
METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "R2",
    "MAPE",
    "Acc_2yr",
    "Acc_5yr",
    "Acc_10yr",
    "Max_Error",
    "Median_Error",
    "Min_Error",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run camera-ready revision analyses without editing the paper.")
    parser.add_argument("--data-csv", type=Path, default=root / "data" / "NewAgeSplit.csv")
    parser.add_argument("--cnn-test-preds", type=Path, default=root / "01_CNN_Ensemble" / "predictions" / "test_image_predictions.csv")
    parser.add_argument("--cnn-val-preds", type=Path, default=root / "01_CNN_Ensemble" / "predictions" / "val_image_predictions.csv")
    parser.add_argument("--vit-test-preds", type=Path, default=root / "04_ViT_Ensemble" / "predictions" / "test_image_predictions.csv")
    parser.add_argument("--vit-val-preds", type=Path, default=root / "04_ViT_Ensemble" / "predictions" / "val_image_predictions.csv")
    parser.add_argument("--dit-preds-dir", type=Path, default=root / "results" / "experiment_06")
    parser.add_argument("--dit-test-preds", type=Path, default=None)
    parser.add_argument("--dit-val-preds", type=Path, default=None)
    parser.add_argument("--cv-preds-dir", type=Path, default=root / "05_ViT_CrossVal" / "new_results")
    parser.add_argument("--output-dir", type=Path, default=root / "revisions" / "analysis_outputs")
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--figure-format", choices=["png", "pdf", "both"], default="both")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_cell(value: object, float_format: str) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, (float, np.floating)):
        return float_format % float(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return latex_escape(value)


def save_latex(df: pd.DataFrame, path: Path, caption: str, label: str, float_format: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "l" + "r" * max(0, len(df.columns) - 1)
    latex_newline = " " + chr(92) * 2
    table_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + latex_newline,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        table_lines.append(" & ".join(latex_cell(row[c], float_format) for c in df.columns) + latex_newline)
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")


def load_labels(data_csv: Path, blocked: list[str]) -> pd.DataFrame:
    if not data_csv.is_file():
        blocked.append(f"BLOCKED: missing label file: {data_csv}")
        return pd.DataFrame(columns=["File", "Set", "Age", "WriterNumber"])
    labels = pd.read_csv(data_csv)
    needed = {"File", "Set", "Age"}
    missing = sorted(needed - set(labels.columns))
    if missing:
        blocked.append(f"BLOCKED: label file {data_csv} is missing columns: {missing}")
    if "WriterNumber" not in labels.columns:
        labels["WriterNumber"] = labels["File"].astype(str).str.extract(r"w(\d+)").astype(float)
    return labels


def test_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    return labels[labels["Set"].astype(str).str.lower() == "test"].copy()


def true_age_dict(labels: pd.DataFrame) -> dict[str, float]:
    if labels.empty:
        return {}
    return dict(zip(labels["File"].astype(str), labels["Age"].astype(float)))


def age_group(age: float) -> str:
    if age <= 15:
        return "<=15"
    if age <= 25:
        return "16-25"
    if age <= 50:
        return "26-50"
    return ">50"


def age_bin(age: float) -> str | None:
    for lo, hi in AGE_BINS:
        if lo <= age < hi:
            return f"[{lo},{hi})"
    return None


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    errors = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = 0.0 if np.isnan(mape) else float(mape)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape,
        "Acc_2yr": float(np.mean(errors <= 2) * 100),
        "Acc_5yr": float(np.mean(errors <= 5) * 100),
        "Acc_10yr": float(np.mean(errors <= 10) * 100),
        "Max_Error": float(np.max(errors)),
        "Median_Error": float(np.median(errors)),
        "Min_Error": float(np.min(errors)),
    }


def normalize_prediction_table(df: pd.DataFrame, system_name: str, labels_by_file: dict[str, float]) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    id_col = lower.get("unit_id") or lower.get("imageid") or lower.get("file")
    pred_col = lower.get("pred_age") or lower.get("predage") or lower.get("prediction") or lower.get("ensemble")
    true_col = lower.get("true_age") or lower.get("trueage") or lower.get("age")
    if id_col is None or pred_col is None:
        raise ValueError(f"Could not infer id/prediction columns for {system_name}")
    out = pd.DataFrame({"unit_id": df[id_col].astype(str), "pred_age": pd.to_numeric(df[pred_col], errors="coerce")})
    if true_col is not None:
        out["true_age"] = pd.to_numeric(df[true_col], errors="coerce")
    else:
        out["true_age"] = out["unit_id"].map(labels_by_file)
    out["system"] = system_name
    return out.dropna(subset=["true_age", "pred_age"])[["system", "unit_id", "true_age", "pred_age"]].reset_index(drop=True)


def wide_systems_from_file(path: Path, model_names: list[str], labels_by_file: dict[str, float], blocked: list[str], label: str) -> dict[str, pd.DataFrame]:
    if not path.is_file():
        blocked.append(f"BLOCKED: missing {label} prediction file: {path}")
        return {}
    df = pd.read_csv(path)
    id_col = "ImageID" if "ImageID" in df.columns else "unit_id" if "unit_id" in df.columns else "File" if "File" in df.columns else None
    if id_col is None:
        blocked.append(f"BLOCKED: {label} prediction file has no ImageID/unit_id/File column: {path}")
        return {}
    systems = {}
    for model in model_names:
        if model in df.columns:
            sub = pd.DataFrame({"unit_id": df[id_col].astype(str), "pred_age": pd.to_numeric(df[model], errors="coerce")})
            if "TrueAge" in df.columns:
                sub["true_age"] = pd.to_numeric(df["TrueAge"], errors="coerce")
            elif "true_age" in df.columns:
                sub["true_age"] = pd.to_numeric(df["true_age"], errors="coerce")
            else:
                sub["true_age"] = sub["unit_id"].map(labels_by_file)
            sub["system"] = model
            systems[model] = sub.dropna(subset=["true_age", "pred_age"])[["system", "unit_id", "true_age", "pred_age"]].reset_index(drop=True)
    return systems


def read_wide(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def weighted_prediction_table(df: pd.DataFrame, system_name: str, group_models: list[str], weights: dict[str, float], labels_by_file: dict[str, float]) -> pd.DataFrame:
    id_col = "ImageID" if "ImageID" in df.columns else "unit_id" if "unit_id" in df.columns else "File"
    sub = df[[id_col] + group_models].copy()
    preds = np.zeros(len(sub), dtype=float)
    denom = np.zeros(len(sub), dtype=float)
    for model in group_models:
        values = pd.to_numeric(sub[model], errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(values)
        preds[ok] += values[ok] * weights[model]
        denom[ok] += weights[model]
    pred_age = np.divide(preds, denom, out=np.full(len(sub), np.nan), where=denom > 0)
    out = pd.DataFrame({"system": system_name, "unit_id": sub[id_col].astype(str), "pred_age": pred_age})
    if "TrueAge" in df.columns:
        out["true_age"] = pd.to_numeric(df["TrueAge"], errors="coerce")
    elif "true_age" in df.columns:
        out["true_age"] = pd.to_numeric(df["true_age"], errors="coerce")
    else:
        out["true_age"] = out["unit_id"].map(labels_by_file)
    return out.dropna(subset=["true_age", "pred_age"])[["system", "unit_id", "true_age", "pred_age"]].reset_index(drop=True)


def mae_for_model(df: pd.DataFrame, model: str, labels_by_file: dict[str, float]) -> float:
    id_col = "ImageID" if "ImageID" in df.columns else "unit_id" if "unit_id" in df.columns else "File"
    sub = df[[id_col, model]].dropna().copy()
    y_true = sub[id_col].astype(str).map(labels_by_file).to_numpy(dtype=float)
    y_pred = pd.to_numeric(sub[model], errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(mean_absolute_error(y_true[ok], y_pred[ok]))


def grid_weights(val_df: pd.DataFrame, group_models: list[str], labels_by_file: dict[str, float], step: float = 0.1) -> tuple[dict[str, float], float]:
    candidates = np.arange(step, 1.0, step)
    best_weights = None
    best_mae = float("inf")
    import itertools

    for combo in itertools.product(candidates, repeat=len(group_models)):
        if not np.isclose(sum(combo), 1.0, atol=1e-8):
            continue
        weights = dict(zip(group_models, map(float, combo)))
        pred_df = weighted_prediction_table(val_df, "tmp", group_models, weights, labels_by_file)
        if pred_df.empty:
            continue
        mae = float(mean_absolute_error(pred_df["true_age"], pred_df["pred_age"]))
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
    if best_weights is None:
        raise ValueError("No valid grid-search weights found")
    return best_weights, best_mae


def load_cnn_systems(args: argparse.Namespace, labels_by_file: dict[str, float], blocked: list[str], metadata: dict) -> dict[str, pd.DataFrame]:
    systems = wide_systems_from_file(args.cnn_test_preds, CNN_MODEL_NAMES, labels_by_file, blocked, "CNN test")
    test_df = read_wide(args.cnn_test_preds)
    val_df = read_wide(args.cnn_val_preds)
    if test_df is not None and val_df is not None and all(m in test_df.columns for m in CNN_BEST3_MODELS) and all(m in val_df.columns for m in CNN_BEST3_MODELS):
        weights, val_mae = grid_weights(val_df, CNN_BEST3_MODELS, labels_by_file)
        systems["CNN_Best3Grid"] = weighted_prediction_table(test_df, "CNN_Best3Grid", CNN_BEST3_MODELS, weights, labels_by_file)
        metadata["CNN_Best3Grid_weights"] = weights
        metadata["CNN_Best3Grid_val_mae"] = val_mae
    else:
        blocked.append(f"BLOCKED: cannot compute CNN_Best3Grid; missing columns or validation file: {args.cnn_val_preds}")
    return systems


def load_vit_systems(args: argparse.Namespace, labels_by_file: dict[str, float], blocked: list[str], metadata: dict) -> dict[str, pd.DataFrame]:
    systems = wide_systems_from_file(args.vit_test_preds, VIT_MODEL_NAMES, labels_by_file, blocked, "official ViT test")
    test_df = read_wide(args.vit_test_preds)
    if test_df is not None and all(m in test_df.columns for m in VIT_MODEL_NAMES):
        weights = {m: 1.0 / len(VIT_MODEL_NAMES) for m in VIT_MODEL_NAMES}
        systems["ViT_EqualMean"] = weighted_prediction_table(test_df, "ViT_EqualMean", VIT_MODEL_NAMES, weights, labels_by_file)
        metadata["ViT_EqualMean_weights"] = weights
    return systems


def infer_dit_files(preds_dir: Path, split: str) -> dict[str, Path]:
    return {name: preds_dir / f"{stem}_{split}_preds.csv" for name, stem in DIT_FILE_STEMS.items()}


def dit_wide_from_dir(preds_dir: Path, split: str, labels_by_file: dict[str, float], blocked: list[str]) -> pd.DataFrame | None:
    files = infer_dit_files(preds_dir, split)
    dfs = []
    for name, path in files.items():
        if not path.is_file():
            blocked.append(f"BLOCKED: missing DiT {split} prediction file for {name}: {path}")
            continue
        try:
            raw = pd.read_csv(path)
            norm = normalize_prediction_table(raw, name, labels_by_file)
            dfs.append(norm[["unit_id", "pred_age"]].rename(columns={"pred_age": name}))
        except Exception as exc:
            blocked.append(f"BLOCKED: failed to read DiT {split} predictions for {name}: {path} ({exc})")
    if not dfs:
        return None
    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="unit_id", how="outer")
    out["ImageID"] = out["unit_id"]
    out["TrueAge"] = out["unit_id"].map(labels_by_file)
    return out.drop(columns=["unit_id"])


def load_dit_systems(args: argparse.Namespace, labels_by_file: dict[str, float], blocked: list[str], metadata: dict) -> dict[str, pd.DataFrame]:
    systems: dict[str, pd.DataFrame] = {}
    if args.dit_test_preds is not None:
        systems.update(wide_systems_from_file(args.dit_test_preds, list(DIT_FILE_STEMS), labels_by_file, blocked, "DiT test"))
        test_df = read_wide(args.dit_test_preds)
    else:
        test_df = dit_wide_from_dir(args.dit_preds_dir, "test", labels_by_file, blocked)
        if test_df is not None:
            systems.update(wide_systems_from_file_from_df(test_df, list(DIT_FILE_STEMS), labels_by_file))
    if args.dit_val_preds is not None:
        val_df = read_wide(args.dit_val_preds)
    else:
        val_df = dit_wide_from_dir(args.dit_preds_dir, "val", labels_by_file, blocked)
    if test_df is None or val_df is None:
        blocked.append("BLOCKED: cannot compute DiT_Best2Grid; missing DiT official-split val/test predictions")
        return systems
    available = [m for m in DIT_FILE_STEMS if m in test_df.columns and m in val_df.columns]
    if len(available) < 2:
        blocked.append("BLOCKED: cannot compute DiT_Best2Grid; fewer than two DiT models are available")
        return systems
    val_rank = sorted(available, key=lambda m: mae_for_model(val_df, m, labels_by_file))
    group = val_rank[:2]
    weights, val_mae = grid_weights(val_df, group, labels_by_file)
    systems["DiT_Best2Grid"] = weighted_prediction_table(test_df, "DiT_Best2Grid", group, weights, labels_by_file)
    metadata["DiT_Best2Grid_models"] = group
    metadata["DiT_Best2Grid_weights"] = weights
    metadata["DiT_Best2Grid_val_mae"] = val_mae
    return systems


def wide_systems_from_file_from_df(df: pd.DataFrame, model_names: list[str], labels_by_file: dict[str, float]) -> dict[str, pd.DataFrame]:
    systems = {}
    id_col = "ImageID" if "ImageID" in df.columns else "unit_id" if "unit_id" in df.columns else "File"
    for model in model_names:
        if model not in df.columns:
            continue
        sub = pd.DataFrame({"unit_id": df[id_col].astype(str), "pred_age": pd.to_numeric(df[model], errors="coerce")})
        if "TrueAge" in df.columns:
            sub["true_age"] = pd.to_numeric(df["TrueAge"], errors="coerce")
        elif "true_age" in df.columns:
            sub["true_age"] = pd.to_numeric(df["true_age"], errors="coerce")
        else:
            sub["true_age"] = sub["unit_id"].map(labels_by_file)
        sub["system"] = model
        systems[model] = sub.dropna(subset=["true_age", "pred_age"])[["system", "unit_id", "true_age", "pred_age"]].reset_index(drop=True)
    return systems


def metric_table(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in systems.items():
        if df.empty:
            continue
        row = {"system": name, "n": len(df)}
        row.update(compute_metrics(df["true_age"], df["pred_age"]))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["system", "n"] + METRIC_COLUMNS)
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str, b: int, rng: np.random.Generator) -> tuple[float, float, float, np.ndarray]:
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2
    stats = np.empty(b, dtype=float)
    n = len(y_true)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        if metric == "MAE":
            stats[i] = float(abs_errors[idx].mean())
        elif metric == "RMSE":
            stats[i] = float(np.sqrt(sq_errors[idx].mean()))
        else:
            raise ValueError(metric)
    point = float(abs_errors.mean()) if metric == "MAE" else float(np.sqrt(sq_errors.mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi), stats


def run_bootstrap(systems: dict[str, pd.DataFrame], output_dir: Path, b: int) -> pd.DataFrame:
    rows = []
    raw_dir = output_dir / "bootstrap_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for system, df in systems.items():
        y_true = df["true_age"].to_numpy(dtype=float)
        y_pred = df["pred_age"].to_numpy(dtype=float)
        for metric in ["MAE", "RMSE"]:
            rng = np.random.default_rng(SEED)
            point, lo, hi, stats = bootstrap_metric(y_true, y_pred, metric, b, rng)
            safe_system = system.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            np.save(raw_dir / f"{safe_system}_{metric}_bootstrap.npy", stats)
            rows.append({"system": system, "metric": metric, "point": point, "ci_low": lo, "ci_high": hi, "B": b})
    return pd.DataFrame(rows)


def aligned_errors(system_a: pd.DataFrame, system_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    a = system_a[["unit_id", "true_age", "pred_age"]].rename(columns={"pred_age": "pred_a"})
    b = system_b[["unit_id", "true_age", "pred_age"]].rename(columns={"pred_age": "pred_b"})
    joined = a.merge(b, on=["unit_id", "true_age"], how="inner").sort_values("unit_id")
    err_a = np.abs(joined["true_age"].to_numpy(dtype=float) - joined["pred_a"].to_numpy(dtype=float))
    err_b = np.abs(joined["true_age"].to_numpy(dtype=float) - joined["pred_b"].to_numpy(dtype=float))
    return err_a, err_b, joined["unit_id"].astype(str).tolist()


def paired_bootstrap_diff(err_a: np.ndarray, err_b: np.ndarray, b: int, rng: np.random.Generator) -> tuple[float, float, float, float, bool, np.ndarray]:
    if err_a.shape != err_b.shape:
        raise ValueError("systems must cover identical units")
    n = err_a.size
    diffs = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(err_a[idx].mean() - err_b[idx].mean())
    point = float(err_a.mean() - err_b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float((diffs > 0).mean()), float((diffs < 0).mean())))
    significant = bool((lo > 0) or (hi < 0))
    return point, float(lo), float(hi), p, significant, diffs


def best_single_dit(metric_df: pd.DataFrame) -> str | None:
    candidates = [name for name in DIT_FILE_STEMS if name in set(metric_df["system"])]
    if not candidates:
        return None
    sub = metric_df[metric_df["system"].isin(candidates)].sort_values("MAE")
    return str(sub.iloc[0]["system"])


def best_single_vit(metric_df: pd.DataFrame) -> str | None:
    candidates = [name for name in VIT_MODEL_NAMES if name in set(metric_df["system"])]
    if not candidates:
        return None
    sub = metric_df[metric_df["system"].isin(candidates)].sort_values("MAE")
    return str(sub.iloc[0]["system"])


def run_paired_bootstrap(systems: dict[str, pd.DataFrame], metric_df: pd.DataFrame, output_dir: Path, b: int, blocked: list[str]) -> pd.DataFrame:
    best_dit = best_single_dit(metric_df)
    comparisons = []
    if "DiT_Best2Grid" in systems and best_dit is not None:
        comparisons.append(("DiT_Best2Grid", best_dit))
    else:
        blocked.append("BLOCKED: paired bootstrap DiT_Best2Grid vs best single DiT needs DiT official predictions")
    if "CNN_Best3Grid" in systems and "ResNet50" in systems:
        comparisons.append(("CNN_Best3Grid", "ResNet50"))
    else:
        blocked.append("BLOCKED: paired bootstrap CNN_Best3Grid vs ResNet50 needs CNN official predictions")
    if "DiT_Best2Grid" in systems and "CNN_Best3Grid" in systems:
        comparisons.append(("DiT_Best2Grid", "CNN_Best3Grid"))
    else:
        blocked.append("BLOCKED: paired bootstrap DiT_Best2Grid vs CNN_Best3Grid needs both systems")
    if best_dit is not None and "CNN_Best3Grid" in systems:
        comparisons.append((best_dit, "CNN_Best3Grid"))
    else:
        blocked.append("BLOCKED: paired bootstrap best single DiT vs CNN_Best3Grid needs best single DiT")
    rows = []
    raw_dir = output_dir / "paired_bootstrap_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for system_a, system_b in comparisons:
        err_a, err_b, unit_ids = aligned_errors(systems[system_a], systems[system_b])
        if len(unit_ids) == 0:
            blocked.append(f"BLOCKED: paired bootstrap {system_a} vs {system_b} has zero aligned units")
            continue
        rng = np.random.default_rng(SEED)
        point, lo, hi, p, significant, diffs = paired_bootstrap_diff(err_a, err_b, b, rng)
        safe = f"{system_a}_vs_{system_b}".replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        np.save(raw_dir / f"{safe}_diffs.npy", diffs)
        rows.append({
            "system_A": system_a,
            "system_B": system_b,
            "delta_MAE_A_minus_B": point,
            "ci_low": lo,
            "ci_high": hi,
            "p_bootstrap": p,
            "significant": significant,
            "n_aligned": len(unit_ids),
            "B": b,
        })
    return pd.DataFrame(rows)


def test_distribution(labels: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    test_df = test_labels(labels)
    if test_df.empty:
        return pd.DataFrame(), "BLOCKED: cannot summarize official test distribution because labels are unavailable."
    rows = []
    n = len(test_df)
    for lo, hi in AGE_BINS:
        count = int(((test_df["Age"] >= lo) & (test_df["Age"] < hi)).sum())
        rows.append({"type": "fig3_bin", "bin": f"[{lo},{hi})", "count": count, "percent": 100.0 * count / n})
    for group in AGE_GROUPS:
        count = int(test_df["Age"].apply(age_group).eq(group).sum())
        rows.append({"type": "error_group", "bin": group, "count": count, "percent": 100.0 * count / n})
    young = int((test_df["Age"] <= 25).sum())
    old = int((test_df["Age"] > 50).sum())
    summary = f"Official test split size is {n}. Writers/pages with age <=25: {young} ({100.0 * young / n:.1f}%). Writers/pages with age >50: {old} ({100.0 * old / n:.1f}%)."
    return pd.DataFrame(rows), summary


def error_by_age_group(systems: dict[str, pd.DataFrame], system_names: list[str]) -> pd.DataFrame:
    rows = []
    for system in system_names:
        df = systems.get(system)
        if df is None or df.empty:
            continue
        work = df.copy()
        work["age_group"] = work["true_age"].apply(age_group)
        for group in AGE_GROUPS:
            sub = work[work["age_group"] == group]
            if sub.empty:
                rows.append({"system": system, "age_group": group, "n": 0, "MAE": np.nan, "RMSE": np.nan})
                continue
            err = sub["true_age"].to_numpy(dtype=float) - sub["pred_age"].to_numpy(dtype=float)
            rows.append({
                "system": system,
                "age_group": group,
                "n": len(sub),
                "MAE": float(np.abs(err).mean()),
                "RMSE": float(np.sqrt((err ** 2).mean())),
            })
    return pd.DataFrame(rows)


def to_bin(age: float) -> int:
    if age <= 15:
        return 0
    if age <= 25:
        return 1
    if age <= 50:
        return 2
    return 3


def binned_classification(systems: dict[str, pd.DataFrame], system_names: list[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    labels = AGE_GROUPS
    for system in system_names:
        df = systems.get(system)
        if df is None or df.empty:
            continue
        true_bins = np.array([to_bin(x) for x in df["true_age"]])
        pred_bins = np.array([to_bin(x) for x in df["pred_age"]])
        overall = float(np.mean(true_bins == pred_bins) * 100)
        row = {"system": system, "overall_accuracy": overall, "n": len(df)}
        matrix = np.zeros((4, 4), dtype=int)
        for t, p in zip(true_bins, pred_bins):
            matrix[t, p] += 1
        for i, label in enumerate(labels):
            denom = matrix[i].sum()
            row[f"recall_{label}"] = float(matrix[i, i] / denom * 100) if denom else np.nan
        cm_df = pd.DataFrame(matrix, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        safe = system.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        save_csv(cm_df.reset_index().rename(columns={"index": "true_bin"}), output_dir / f"confusion_matrix_{safe}.csv")
        rows.append(row)
    return pd.DataFrame(rows)


def cv_per_fold_table(cv_preds_dir: Path, labels_by_file: dict[str, float], blocked: list[str]) -> pd.DataFrame:
    if not cv_preds_dir.is_dir():
        blocked.append(f"BLOCKED: missing CV predictions directory: {cv_preds_dir}")
        return pd.DataFrame()
    rows = []
    for path in sorted(cv_preds_dir.glob("*_fold*_preds.csv")):
        raw = pd.read_csv(path)
        needed = {"Model", "Fold", "ImageID", "Prediction"}
        if not needed.issubset(raw.columns):
            blocked.append(f"BLOCKED: CV prediction file has unexpected columns: {path}")
            continue
        raw["true_age"] = raw["ImageID"].astype(str).map(labels_by_file)
        raw = raw.dropna(subset=["true_age", "Prediction"])
        if raw.empty:
            continue
        metrics = compute_metrics(raw["true_age"], raw["Prediction"])
        rows.append({"Model": str(raw.iloc[0]["Model"]), "Fold": int(raw.iloc[0]["Fold"]), **metrics})
    if not rows:
        blocked.append(f"BLOCKED: no usable CV fold prediction CSVs found under {cv_preds_dir}")
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Model", "Fold"]).reset_index(drop=True)


def plot_figure6(error_df: pd.DataFrame, output_dir: Path, figure_format: str) -> list[Path]:
    if error_df.empty:
        return []
    plot_df = error_df.dropna(subset=["MAE"]).copy()
    systems = list(plot_df["system"].drop_duplicates())
    groups = AGE_GROUPS
    x = np.arange(len(groups))
    width = min(0.8 / max(1, len(systems)), 0.18)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    offsets = (np.arange(len(systems)) - (len(systems) - 1) / 2) * width
    for offset, system in zip(offsets, systems):
        vals = []
        for group in groups:
            sub = plot_df[(plot_df["system"] == system) & (plot_df["age_group"] == group)]
            vals.append(float(sub.iloc[0]["MAE"]) if not sub.empty else np.nan)
        ax.bar(x + offset, vals, width=width, label=system)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("Age group")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Age-group MAE on the official HHD test split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = []
    if figure_format in {"png", "both"}:
        path = output_dir / "figure6_age_group_mae.png"
        fig.savefig(path, dpi=300)
        paths.append(path)
    if figure_format in {"pdf", "both"}:
        path = output_dir / "figure6_age_group_mae.pdf"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def raw_error_rows(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for system, df in systems.items():
        out = df.copy()
        out["error"] = out["pred_age"] - out["true_age"]
        out["abs_error"] = out["error"].abs()
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["system", "unit_id", "true_age", "pred_age", "error", "abs_error"])
    return pd.concat(rows, ignore_index=True)


def report_inputs(args: argparse.Namespace, blocked: list[str], metadata: dict) -> str:
    paths = {
        "labels": args.data_csv,
        "cnn_test_predictions": args.cnn_test_preds,
        "cnn_val_predictions": args.cnn_val_preds,
        "vit_test_predictions": args.vit_test_preds,
        "vit_val_predictions": args.vit_val_preds,
        "dit_predictions_directory": args.dit_preds_dir,
        "cv_predictions_directory": args.cv_preds_dir,
    }
    lines = ["# Camera-ready revision analysis report", "", "## Inputs", ""]
    for name, path in paths.items():
        exists = path.exists()
        lines.append(f"- {name}: `{path}` ({'FOUND' if exists else 'MISSING'})")
    lines.extend(["", "## Metadata", "", "```json", json.dumps(metadata, indent=2, sort_keys=True), "```", "", "## Blocked notes", ""])
    if blocked:
        lines.extend(f"- {item}" for item in blocked)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocked: list[str] = []
    metadata: dict = {"SEED": SEED, "bootstrap_resamples": 0 if args.skip_bootstrap else args.bootstrap_resamples}
    labels = load_labels(args.data_csv, blocked)
    labels_by_file = true_age_dict(labels)
    distribution_df, distribution_summary = test_distribution(labels)
    if not distribution_df.empty:
        save_csv(distribution_df, args.output_dir / "test_distribution.csv")
    write_text(args.output_dir / "test_distribution_summary.txt", distribution_summary + "\n")
    systems = {}
    systems.update(load_cnn_systems(args, labels_by_file, blocked, metadata))
    systems.update(load_vit_systems(args, labels_by_file, blocked, metadata))
    systems.update(load_dit_systems(args, labels_by_file, blocked, metadata))
    raw_errors = raw_error_rows(systems)
    save_csv(raw_errors, args.output_dir / "per_sample_errors.csv")
    metrics = metric_table(systems)
    save_csv(metrics, args.output_dir / "single_models_official_split.csv")
    if not metrics.empty:
        save_latex(metrics, args.output_dir / "single_models_official_split.tex", "Official-split page-level metrics for available systems.", "tab:single-models-official")
    headline_systems = [s for s in ["CNN_Best3Grid", "DiT_Best2Grid", best_single_dit(metrics), best_single_vit(metrics), "ViT_EqualMean"] if s]
    if "CNN_Best3Grid" not in systems:
        blocked.append("BLOCKED: age-group table/figure missing CNN_Best3Grid")
    if "DiT_Best2Grid" not in systems:
        blocked.append("BLOCKED: age-group table/figure missing DiT_Best2Grid")
    if best_single_vit(metrics) is None:
        blocked.append("BLOCKED: Figure 6 cannot include ViT until official-split ViT predictions are available")
    age_group_df = error_by_age_group(systems, headline_systems)
    save_csv(age_group_df, args.output_dir / "error_by_age_group.csv")
    if not age_group_df.empty:
        save_latex(age_group_df, args.output_dir / "error_by_age_group.tex", "Age-group error analysis on the official HHD test split.", "tab:error-by-age-group")
    figure_paths = plot_figure6(age_group_df, args.output_dir, args.figure_format)
    metadata["figure6_outputs"] = [str(p) for p in figure_paths]
    class_df = binned_classification(systems, headline_systems, args.output_dir)
    save_csv(class_df, args.output_dir / "binned_classification.csv")
    if not class_df.empty:
        save_latex(class_df, args.output_dir / "binned_classification.tex", "Binned classification accuracy from continuous age predictions.", "tab:binned-classification")
    cv_df = cv_per_fold_table(args.cv_preds_dir, labels_by_file, blocked)
    save_csv(cv_df, args.output_dir / "cv_per_fold.csv")
    if not cv_df.empty:
        save_latex(cv_df, args.output_dir / "cv_per_fold.tex", "Per-fold cross-validation metrics for available ViT predictions.", "tab:cv-per-fold")
    if not args.skip_bootstrap and systems:
        cis = run_bootstrap({k: systems[k] for k in headline_systems if k in systems}, args.output_dir, args.bootstrap_resamples)
        save_csv(cis, args.output_dir / "bootstrap_cis.csv")
        if not cis.empty:
            save_latex(cis, args.output_dir / "bootstrap_cis.tex", "Percentile bootstrap confidence intervals on official-split metrics.", "tab:bootstrap-cis")
        paired = run_paired_bootstrap(systems, metrics, args.output_dir, args.bootstrap_resamples, blocked)
        save_csv(paired, args.output_dir / "paired_bootstrap.csv")
        if not paired.empty:
            save_latex(paired, args.output_dir / "paired_bootstrap.tex", "Paired bootstrap differences in MAE on the official HHD test split.", "tab:paired-bootstrap")
    elif args.skip_bootstrap:
        blocked.append("BLOCKED: bootstrap CI and paired-bootstrap outputs skipped by --skip-bootstrap")
    write_text(args.output_dir / "run_report.md", report_inputs(args, blocked, metadata))
    print(f"Saved revision analysis outputs to {args.output_dir}")
    if blocked:
        print(f"Completed with {len(blocked)} BLOCKED note(s). See {args.output_dir / 'run_report.md'}")


if __name__ == "__main__":
    main()
