#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TM110 – CTG NSP Inference (bare .pkl loader)

Assumes the saved file is a plain scikit-learn estimator (not a dict artifact).
The script:
- Reads .xlsx/.xls/.csv
- Validates & orders the 24 required features
- Predicts NSP (1/2/3) and prints/saves results
"""

import argparse, os, sys
import pandas as pd
import numpy as np
import joblib

REQUIRED = [
    "b","e","AC","FM","UC","DL","DS","DP","DR","LB",
    "ASTV","MSTV","ALTV","MLTV","Width","Min","Max",
    "Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"
]

CLASS_NAMES = {1:"Normal", 2:"Suspect", 3:"Pathologic"}

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv.")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Case-insensitive column matching, enforce REQUIRED order
    lower = {c.lower().strip(): c for c in df.columns}
    missing, chosen = [], []
    for f in REQUIRED:
        c = lower.get(f.lower())
        if c is None: missing.append(f)
        else: chosen.append(c)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nGot columns: " + ", ".join(map(str, df.columns))
        )
    X = df[chosen].copy()
    X.columns = REQUIRED  # set canonical names/order
    return X

def main():
    p = argparse.ArgumentParser(description="TM110 – CTG NSP Inference")
    p.add_argument("--input", required=True, help="Path to .xlsx/.xls/.csv with features")
    p.add_argument("--model", default="models/HGB_NSP.pkl", help="Path to plain sklearn .pkl")
    p.add_argument("--output", default="predictions.csv", help="Where to save predictions CSV")
    args = p.parse_args()

    # Load input
    try:
        df_raw = read_any(args.input)
    except Exception as e:
        print(f"[Error] Could not read input: {e}", file=sys.stderr); sys.exit(2)
    if df_raw.empty:
        print("[Error] Input has no rows.", file=sys.stderr); sys.exit(2)

    # Align columns
    try:
        X = normalize(df_raw)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr); sys.exit(2)

    # Load model (bare estimator)
    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(f"[Error] Could not load model: {e}", file=sys.stderr); sys.exit(2)

    # Predict
    try:
        y = model.predict(X)
    except Exception as e:
        print(f"[Error] Prediction failed: {e}", file=sys.stderr); sys.exit(2)

    out = pd.DataFrame({
        "row_id": np.arange(len(X)),
        "pred_int": y
    })
    out["pred_label"] = out["pred_int"].map(CLASS_NAMES)

    # Try probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            classes = getattr(model, "classes_", None)
            if classes is not None:
                cols = [f"proba_{int(c)}" for c in classes]
            else:
                cols = [f"proba_{i}" for i in range(proba.shape[1])]
            out = pd.concat([out, pd.DataFrame(proba, columns=cols)], axis=1)
        except Exception:
            pass

    # Save & preview
    try:
        out.to_csv(args.output, index=False)
    except Exception as e:
        print(f"[Error] Could not save output CSV: {e}", file=sys.stderr); sys.exit(2)

    print("=== TM110 – NSP Predictions (head) ===")
    print(out.head(10).to_string(index=False))
    print(f"\nSaved predictions to: {os.path.abspath(args.output)}")
    print("Legend: 1=Normal, 2=Suspect, 3=Pathologic")

if __name__ == "__main__":
    main()
