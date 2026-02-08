import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_float(x):
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _format_float(x: float, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        if math.isnan(float(x)):
            return ""
    except Exception:
        return ""
    return f"{float(x):.{digits}f}"


def _latex_escape(text: str) -> str:
    # Minimal escaping for LaTeX table cells
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _build_split(df: pd.DataFrame, seed: int):
    from sklearn.model_selection import train_test_split

    text_column = "combined_text" if "combined_text" in df.columns else "clean_text"
    x_all = df[text_column].astype(str).values
    y_all = df["label"].astype(str).values

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all,
        y_all,
        test_size=0.30,
        random_state=seed,
        stratify=y_all,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )
    return x_train, y_train, x_test, y_test


def _load_transformer(
    model_dir: Path,
    *,
    checkpoint_hint: str | None = None,
    num_labels: int | None = None,
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Some saved final_model_dir folders were created without config.json.
    # In that case we reconstruct config from the original checkpoint name.
    config_override = None
    config_path = model_dir / "config.json"
    if not config_path.exists() and checkpoint_hint:
        config_override = AutoConfig.from_pretrained(str(checkpoint_hint))

    if config_override is not None and (num_labels is not None) and (label2id is not None) and (id2label is not None):
        config_override.num_labels = int(num_labels)
        config_override.label2id = {str(k): int(v) for k, v in label2id.items()}
        config_override.id2label = {int(k): str(v) for k, v in id2label.items()}

    try:
        if config_override is not None:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config_override)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except RuntimeError as e:
        # Some saved runs may have a stale config.json with num_labels=2 while the checkpoint head is 5.
        # If caller provides label metadata, retry with an overridden config.
        if num_labels is None or label2id is None or id2label is None:
            raise
        msg = str(e)
        if "size mismatch" not in msg:
            raise

        # Retry with an overridden config from either the local folder or the checkpoint hint.
        if config_override is not None:
            config = config_override
        elif checkpoint_hint:
            config = AutoConfig.from_pretrained(str(checkpoint_hint))
        else:
            config = AutoConfig.from_pretrained(model_dir)
        config.num_labels = int(num_labels)
        config.label2id = {str(k): int(v) for k, v in label2id.items()}
        config.id2label = {int(k): str(v) for k, v in id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _batched_predict(tokenizer, model, device, texts, *, batch_size: int, max_length: int):
    import torch

    pred_ids = []
    probs_all = []

    # Respect tokenizer/model max length constraints
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is None or tok_max == int(1e30):
        tok_max = 512
    max_length = int(min(max_length, tok_max, 512))

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

        pred_ids.extend(pred.detach().cpu().numpy().tolist())
        probs_all.append(probs.detach().cpu().numpy())

    probs_np = np.concatenate(probs_all, axis=0)
    return np.array(pred_ids, dtype=int), probs_np


def _compute_advanced_metrics(y_true_ids: np.ndarray, y_pred_ids: np.ndarray, y_prob: np.ndarray | None):
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    accuracy = float(accuracy_score(y_true_ids, y_pred_ids))
    error_rate = float(1.0 - accuracy)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_ids,
        y_pred_ids,
        average="macro",
        zero_division=0,
    )

    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true_ids,
        y_pred_ids,
        average="weighted",
        zero_division=0,
    )

    kappa = float(cohen_kappa_score(y_true_ids, y_pred_ids))
    mcc = float(matthews_corrcoef(y_true_ids, y_pred_ids))

    roc_auc_macro = float("nan")
    roc_auc_weighted = float("nan")

    if y_prob is not None:
        try:
            roc_auc_macro = float(
                roc_auc_score(y_true_ids, y_prob, multi_class="ovr", average="macro")
            )
            roc_auc_weighted = float(
                roc_auc_score(y_true_ids, y_prob, multi_class="ovr", average="weighted")
            )
        except ValueError:
            # Can happen if a class is missing in y_true for the split
            roc_auc_macro = float("nan")
            roc_auc_weighted = float("nan")

    return {
        "Accuracy": accuracy,
        "Precision_Macro": float(precision),
        "Recall_Macro": float(recall),
        "F1_Macro": float(f1),
        "F1_Weighted": float(f1_weighted),
        "Cohen_Kappa": kappa,
        "MCC": mcc,
        "ROC_AUC_Macro": roc_auc_macro,
        "ROC_AUC_Weighted": roc_auc_weighted,
        "Error_Rate": error_rate,
    }


def _write_table1_tex(df: pd.DataFrame, out_tex: Path):
    row_end = r"\\"
    lines = []
    lines.append("\\begin{table}")
    lines.append("\\caption{Standard Classification Performance Metrics}")
    lines.append("\\label{tab:standard_metrics}")
    lines.append("\\begin{tabular}{clccccc}")
    lines.append("\\toprule")
    lines.append(
        f"Rank & Model & Accuracy & Precision & Recall & F1-Score (Macro) & F1-Score (Weighted) {row_end}"
    )
    lines.append("\\midrule")

    for _, r in df.iterrows():
        rank = int(r["Rank"])
        model = _latex_escape(r["Model"])
        acc = _format_float(r["Accuracy"], 4)
        prec = _format_float(r["Precision"], 4)
        rec = _format_float(r["Recall"], 4)
        f1m = _format_float(r["F1-Score (Macro)"], 4)
        f1w = _format_float(r["F1-Score (Weighted)"], 4)
        lines.append(f"{rank} & {model} & {acc} & {prec} & {rec} & {f1m} & {f1w} {row_end}")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table5_tex(df: pd.DataFrame, out_tex: Path):
    row_end = r"\\"
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical significance tests using McNemar's test.}")
    lines.append("\\label{tab:significance_tests}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{llrrll}")
    lines.append("\\toprule")
    lines.append(
        f"Model 1 & Model 2 & Statistic & p-value & Significant ($\\alpha=0.05$) & Interpretation {row_end}"
    )
    lines.append("\\midrule")

    for _, r in df.iterrows():
        m1 = _latex_escape(str(r["Model 1"]))
        m2 = _latex_escape(str(r["Model 2"]))
        stat = _format_float(r["Statistic"], 4)
        p_fmt = _format_float(r["p-value"], 4)
        sig = _latex_escape(str(r.get("Significant (α=0.05)", "")))
        interp = _latex_escape(str(r.get("Interpretation", "")))
        lines.append(f"{m1} & {m2} & {stat} & {p_fmt} & {sig} & {interp} {row_end}")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_per_class(y_true_ids: np.ndarray, y_pred_ids: np.ndarray, id2label: dict[int, str]):
    from sklearn.metrics import precision_recall_fscore_support

    labels = sorted(id2label.keys())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_ids,
        y_pred_ids,
        labels=labels,
        zero_division=0,
    )

    rows = []
    for i, label_id in enumerate(labels):
        rows.append(
            {
                "Class": id2label[label_id],
                "Precision": float(precision[i]),
                "Recall": float(recall[i]),
                "F1-Score": float(f1[i]),
                "Support": int(support[i]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["F1-Score", "Support"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def _write_table2_tex(
    df: pd.DataFrame,
    out_tex: Path,
    *,
    caption: str = "Advanced Evaluation Metrics: Cohen's Kappa, MCC, and ROC-AUC",
    label: str = "tab:advanced_metrics",
):
    row_end = r"\\"
    lines = []
    lines.append("\\begin{table}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    # 7 columns: Rank, Model, Kappa, MCC, AUC_macro, AUC_weighted, ErrorRate
    lines.append("\\begin{tabular}{clccccc}")
    lines.append("\\toprule")
    lines.append(f"Rank & Model & Cohen's Kappa & MCC & ROC-AUC (Macro) & ROC-AUC (Weighted) & Error Rate {row_end}")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        rank = int(r["Rank"])
        model = _latex_escape(r["Model"])
        kappa = _format_float(r["Cohen's Kappa"], 4)
        mcc = _format_float(r["MCC"], 4)
        auc_m = _format_float(r["ROC-AUC (Macro)"], 4)
        auc_w = _format_float(r["ROC-AUC (Weighted)"], 4)
        err = _format_float(r["Error Rate"], 4)

        lines.append(f"{rank} & {model} & {kappa} & {mcc} & {auc_m} & {auc_w} & {err} {row_end}")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table4_tex(df: pd.DataFrame, out_tex: Path, model_name: str):
    row_end = r"\\"
    lines = []
    lines.append("\\begin{table}")
    lines.append(f"\\caption{{Per-Class Performance Metrics for {model_name}}}")
    lines.append("\\label{tab:per_class_metrics}")
    lines.append("\\begin{tabular}{clccccc}")
    lines.append("\\toprule")
    lines.append(f"Rank & Class & Precision & Recall & F1-Score & Support {row_end}")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        rank = int(r["Rank"])
        cls = _latex_escape(r["Class"])
        prec = _format_float(r["Precision"], 4)
        rec = _format_float(r["Recall"], 4)
        f1 = _format_float(r["F1-Score"], 4)
        sup = int(r["Support"])
        lines.append(f"{rank} & {cls} & {prec} & {rec} & {f1} & {sup} {row_end}")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(Path("data") / "english_clean.csv"),
        help="Path to data/english_clean.csv",
    )
    parser.add_argument(
        "--comparison",
        default=str(Path("outputs") / "all_models_comparison.csv"),
        help="Path to outputs/all_models_comparison.csv",
    )
    parser.add_argument(
        "--metrics-summary",
        default=str(Path("metrics_tables") / "complete_metrics_summary.csv"),
        help="Path to metrics_tables/complete_metrics_summary.csv (used to keep baseline rows)",
    )
    parser.add_argument(
        "--label-encoder",
        default=str(Path("models") / "bert_models" / "label_encoder.pkl"),
        help="Path to models/bert_models/label_encoder.pkl (used for fallback label mapping)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for train/val/test split (must match training)",
    )
    parser.add_argument(
        "--max-length-default",
        type=int,
        default=128,
        help="Fallback max_length if not in comparison CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transformer inference",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("metrics_tables")),
        help="Output directory for .csv/.tex tables",
    )
    parser.add_argument(
        "--per-class-model",
        default="RoBERTa",
        help="Model name for per-class table (must exist in comparison CSV)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    comparison_path = Path(args.comparison)
    metrics_summary_path = Path(args.metrics_summary)
    label_encoder_path = Path(args.label_encoder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)

    df_data = pd.read_csv(data_path)
    df_comp = pd.read_csv(comparison_path)

    # Fallback label mapping (used if a saved run has stale config.json)
    fallback_num_labels = None
    fallback_label2id = None
    fallback_id2label = None
    if label_encoder_path.exists():
        with open(label_encoder_path, "rb") as f:
            le = pickle.load(f)
        classes = [str(c) for c in le.classes_]
        fallback_num_labels = len(classes)
        fallback_label2id = {c: i for i, c in enumerate(classes)}
        fallback_id2label = {i: c for i, c in enumerate(classes)}

    # Keep baseline rows from existing summary (if present)
    baseline_rows = []
    baseline_rows_std = []
    if metrics_summary_path.exists():
        df_summary = pd.read_csv(metrics_summary_path)
        # Baselines in summary have these names
        baseline_names = {
            "SVM",
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
        }
        for name in baseline_names:
            match = df_summary[df_summary["Model"] == name]
            if not match.empty:
                r = match.iloc[0].to_dict()
                baseline_rows_std.append(
                    {
                        "Model": name,
                        "Accuracy": _safe_float(r.get("Accuracy")),
                        "Precision": _safe_float(r.get("Precision_Macro")),
                        "Recall": _safe_float(r.get("Recall_Macro")),
                        "F1_Macro": _safe_float(r.get("F1_Macro")),
                        "F1_Weighted": _safe_float(r.get("F1_Weighted")),
                    }
                )
                baseline_rows.append(
                    {
                        "Model": name,
                        "Cohen_Kappa": _safe_float(r.get("Cohen_Kappa")),
                        "MCC": _safe_float(r.get("MCC")),
                        "ROC_AUC_Macro": _safe_float(r.get("ROC_AUC_Macro")),
                        "ROC_AUC_Weighted": _safe_float(r.get("ROC_AUC_Weighted")),
                        "Error_Rate": _safe_float(r.get("Error_Rate")),
                    }
                )

    # Compute transformer metrics consistently for all transformer runs with a final_model_dir
    df_transformers = df_comp[df_comp["Type"] == "Transformer"].copy()
    df_transformers = df_transformers[df_transformers["final_model_dir"].notna()].copy()

    x_train_text, y_train_text, x_test, y_test_labels = _build_split(df_data, seed=args.seed)

    transformer_rows = []
    preds_by_run: dict[tuple[str, str], np.ndarray] = {}
    per_class_df = None

    for _, row in df_transformers.iterrows():
        model_name = str(row["Model"])
        run_tag = str(row.get("run_tag", "") or "")
        checkpoint_hint = str(row.get("Checkpoint", "") or "")
        model_dir = Path(str(row["final_model_dir"]))
        max_len = int(_safe_float(row.get("max_length")) or args.max_length_default)

        if not model_dir.exists():
            # Skip silently (keeps script usable even if some artifacts are missing)
            continue

        try:
            tokenizer, model, device = _load_transformer(
                model_dir,
                checkpoint_hint=checkpoint_hint or None,
                num_labels=fallback_num_labels,
                label2id=fallback_label2id,
                id2label=fallback_id2label,
            )
        except Exception as e:
            print(f"⚠️  Skipping model={model_name} run_tag={run_tag}: failed to load ({type(e).__name__}: {e})")
            continue

        # Use label mapping stored in the model config (most robust)
        label2id = {str(k): int(v) for k, v in getattr(model.config, "label2id", {}).items()}
        id2label = {int(k): str(v) for k, v in getattr(model.config, "id2label", {}).items()}

        missing = sorted(set(y_test_labels) - set(label2id.keys()))
        if missing:
            raise ValueError(
                f"Label mismatch for model={model_name} run_tag={run_tag}: missing labels in model mapping: {missing}"
            )

        y_true_ids = np.array([label2id[lbl] for lbl in y_test_labels], dtype=int)
        y_pred_ids, y_prob = _batched_predict(
            tokenizer,
            model,
            device,
            x_test,
            batch_size=args.batch_size,
            max_length=max_len,
        )

        metrics = _compute_advanced_metrics(y_true_ids, y_pred_ids, y_prob)

        # Store predicted labels so downstream tables (e.g., McNemar) can compare on label strings.
        y_pred_labels = np.array([id2label.get(int(i), str(i)) for i in y_pred_ids], dtype=object)
        preds_by_run[(model_name, run_tag)] = y_pred_labels

        transformer_rows.append(
            {
                "Model": model_name,
                "run_tag": run_tag,
                **metrics,
            }
        )

        if model_name == args.per_class_model:
            per_class_df = _compute_per_class(y_true_ids, y_pred_ids, id2label)

    if not transformer_rows:
        raise RuntimeError("No transformer rows were computed. Check comparison CSV and model directories.")

    df_t = pd.DataFrame(transformer_rows)

    # Prefer the best run per model (max MCC) if multiple runs exist
    df_t_best = (
        df_t.sort_values(["Model", "MCC"], ascending=[True, False])
        .groupby("Model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # For standard metrics table we prefer the best per model by F1_Macro
    df_t_best_std = (
        df_t.sort_values(["Model", "F1_Macro"], ascending=[True, False])
        .groupby("Model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # --- Table 5: Statistical significance tests (McNemar) ---
    out_csv5 = out_dir / "table5_significance_tests.csv"
    out_tex5 = out_dir / "table5_significance_tests.tex"

    significance_rows: list[dict[str, object]] = []
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        reference_model = "RoBERTa"
        ref_row = df_t_best_std[df_t_best_std["Model"] == reference_model]
        if not ref_row.empty:
            ref_run = str(ref_row.iloc[0]["run_tag"])
            y_pred_ref = preds_by_run.get((reference_model, ref_run))
        else:
            y_pred_ref = None

        if y_pred_ref is not None:
            y_true_labels_arr = np.array(y_test_labels, dtype=object)

            def _mcnemar_row(model2_name: str, y_pred2_labels: np.ndarray):
                c1 = (y_pred_ref == y_true_labels_arr)
                c2 = (y_pred2_labels == y_true_labels_arr)

                a = int(np.sum(c1 & c2))
                b = int(np.sum(c1 & (~c2)))
                c = int(np.sum((~c1) & c2))
                d = int(np.sum((~c1) & (~c2)))

                contingency = [[a, b], [c, d]]
                res = mcnemar(contingency, exact=False, correction=True)
                p = float(res.pvalue)

                significance_rows.append(
                    {
                        "Model 1": reference_model,
                        "Model 2": model2_name,
                        "Statistic": float(res.statistic),
                        "p-value": p,
                        "Significant (α=0.05)": "Yes" if p < 0.05 else "No",
                        "Interpretation": "Significantly different" if p < 0.05 else "No significant difference",
                    }
                )

            # Compare against other transformers (best run per model by F1_Macro)
            for _, r in df_t_best_std.iterrows():
                model2 = str(r["Model"])
                if model2 == reference_model:
                    continue
                run2 = str(r["run_tag"])
                y_pred2 = preds_by_run.get((model2, run2))
                if y_pred2 is not None:
                    _mcnemar_row(model2, y_pred2)

            # Compare against baselines (retrain with TF-IDF on the same split)
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
                stop_words="english",
            )
            X_train_tfidf = tfidf.fit_transform(x_train_text)
            X_test_tfidf = tfidf.transform(x_test)

            baseline_models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
                "SVM": LinearSVC(
                    C=1.0,
                    max_iter=1000,
                    random_state=args.seed,
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    random_state=args.seed,
                    n_jobs=-1,
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=args.seed,
                ),
            }

            for name, model in baseline_models.items():
                model.fit(X_train_tfidf, y_train_text)
                y_pred2 = np.array(model.predict(X_test_tfidf), dtype=object)
                _mcnemar_row(name, y_pred2)

            if significance_rows:
                df5 = pd.DataFrame(significance_rows)
                df5 = df5.sort_values(["Significant (α=0.05)", "p-value"], ascending=[False, True]).reset_index(
                    drop=True
                )
                df5.to_csv(out_csv5, index=False)
                _write_table5_tex(df5, out_tex5)

    except Exception as e:
        # Table 5 is optional; don't fail the whole script.
        print(f"⚠️  Skipping Table 5 generation: {type(e).__name__}: {e}")

    df_all = pd.concat(
        [
            df_t_best[
                [
                    "Model",
                    "Cohen_Kappa",
                    "MCC",
                    "ROC_AUC_Macro",
                    "ROC_AUC_Weighted",
                    "Error_Rate",
                ]
            ],
            pd.DataFrame(baseline_rows),
        ],
        ignore_index=True,
    )

    # Rank by MCC desc
    df_all = df_all.sort_values("MCC", ascending=False).reset_index(drop=True)
    df_all.insert(0, "Rank", np.arange(1, len(df_all) + 1))

    # Write table2 CSV + TEX
    out_csv2 = out_dir / "table2_advanced_metrics.csv"
    out_tex2 = out_dir / "table2_advanced_metrics.tex"

    df2 = pd.DataFrame(
        {
            "Rank": df_all["Rank"].astype(int),
            "Model": df_all["Model"],
            "Cohen's Kappa": df_all["Cohen_Kappa"].astype(float),
            "MCC": df_all["MCC"].astype(float),
            "ROC-AUC (Macro)": df_all["ROC_AUC_Macro"].astype(float),
            "ROC-AUC (Weighted)": df_all["ROC_AUC_Weighted"].astype(float),
            "Error Rate": df_all["Error_Rate"].astype(float),
        }
    )
    df2.to_csv(out_csv2, index=False)
    _write_table2_tex(df2, out_tex2)

    # Optional compact Table 2 for the paper: only transformer models, top-K rows.
    compact_topk = 6
    transformer_model_names = set(df_t_best["Model"].tolist())
    df2_compact = df2[df2["Model"].isin(transformer_model_names)].copy()
    df2_compact = df2_compact.head(compact_topk).reset_index(drop=True)
    if not df2_compact.empty:
        df2_compact["Rank"] = np.arange(1, len(df2_compact) + 1)
        out_csv2c = out_dir / "table2_advanced_metrics_compact.csv"
        out_tex2c = out_dir / "table2_advanced_metrics_compact.tex"
        df2_compact.to_csv(out_csv2c, index=False)
        _write_table2_tex(
            df2_compact,
            out_tex2c,
            caption="Advanced Evaluation Metrics (Top Transformers)",
            label="tab:advanced_metrics_compact",
        )

    # Write table1 (standard metrics) including lightweight
    out_csv1 = out_dir / "table1_standard_metrics.csv"
    out_tex1 = out_dir / "table1_standard_metrics.tex"

    baseline_df_std = pd.DataFrame(baseline_rows_std)
    if baseline_df_std.empty:
        baseline_df_std = pd.DataFrame(
            columns=["Model", "Accuracy", "Precision", "Recall", "F1_Macro", "F1_Weighted"]
        )

    df_std = pd.concat(
        [
            df_t_best_std[["Model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "F1_Weighted"]].rename(
                columns={"Precision_Macro": "Precision", "Recall_Macro": "Recall"}
            ),
            baseline_df_std,
        ],
        ignore_index=True,
    )

    if df_std.empty:
        raise RuntimeError(
            "No standard-metric rows available (no transformers and no baselines)."
        )
    df_std = df_std.sort_values("F1_Macro", ascending=False).reset_index(drop=True)
    df_std.insert(0, "Rank", np.arange(1, len(df_std) + 1))

    df1 = pd.DataFrame(
        {
            "Rank": df_std["Rank"].astype(int),
            "Model": df_std["Model"],
            "Accuracy": df_std["Accuracy"].astype(float),
            "Precision": df_std["Precision"].astype(float),
            "Recall": df_std["Recall"].astype(float),
            "F1-Score (Macro)": df_std["F1_Macro"].astype(float),
            "F1-Score (Weighted)": df_std["F1_Weighted"].astype(float),
        }
    )
    df1.to_csv(out_csv1, index=False)
    _write_table1_tex(df1, out_tex1)

    # Write table4 (per-class) if requested model was computed
    if per_class_df is not None:
        out_csv4 = out_dir / "table4_per_class_roberta.csv"
        out_tex4 = out_dir / "table4_per_class_roberta.tex"
        per_class_df.to_csv(out_csv4, index=False)
        _write_table4_tex(per_class_df, out_tex4, args.per_class_model)

    # Tiny machine-readable manifest
    manifest = {
        "seed": args.seed,
        "test_size": 0.15,
        "val_size": 0.15,
        "comparison_csv": str(comparison_path),
        "computed_transformers": sorted(df_t_best["Model"].tolist()),
        "table2_csv": str(out_csv2),
        "table2_tex": str(out_tex2),
        "table2_compact_csv": str(out_dir / "table2_advanced_metrics_compact.csv"),
        "table2_compact_tex": str(out_dir / "table2_advanced_metrics_compact.tex"),
        "table1_csv": str(out_csv1),
        "table1_tex": str(out_tex1),
        "table4_csv": str(out_dir / "table4_per_class_roberta.csv"),
        "table4_tex": str(out_dir / "table4_per_class_roberta.tex"),
        "table5_csv": str(out_dir / "table5_significance_tests.csv"),
        "table5_tex": str(out_dir / "table5_significance_tests.tex"),
    }
    (out_dir / "tables_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {out_csv2}")
    print(f"Wrote: {out_tex2}")
    if (out_dir / "table2_advanced_metrics_compact.csv").exists():
        print(f"Wrote: {out_dir / 'table2_advanced_metrics_compact.csv'}")
        print(f"Wrote: {out_dir / 'table2_advanced_metrics_compact.tex'}")
    print(f"Wrote: {out_csv1}")
    print(f"Wrote: {out_tex1}")
    if per_class_df is not None:
        print(f"Wrote: {out_dir / 'table4_per_class_roberta.csv'}")
        print(f"Wrote: {out_dir / 'table4_per_class_roberta.tex'}")
    if (out_dir / "table5_significance_tests.csv").exists():
        print(f"Wrote: {out_dir / 'table5_significance_tests.csv'}")
        print(f"Wrote: {out_dir / 'table5_significance_tests.tex'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
