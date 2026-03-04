import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "clinical_dialogues.csv"
MODELS_DIR = ROOT / "models"

ASPECTS: Dict[str, Dict[str, List[str]]] = {
    "emergency": {
        "symptoms": [
            "crushing chest pain",
            "shortness of breath at rest",
            "slurred speech",
            "one-sided weakness",
            "vomiting blood",
            "new seizure",
            "confusion after fainting",
            "blue lips",
            "severe allergic swelling",
            "uncontrolled heavy bleeding",
        ],
        "duration": ["for 20 minutes", "since this morning", "for 1 hour", "suddenly today"],
        "severity": ["severe", "very severe", "critical", "rapidly worsening"],
        "context": ["after exercise", "while resting", "with sweating", "with dizziness"],
        "doctor": [
            "Go to emergency now.",
            "Call emergency services immediately.",
            "This requires emergency department evaluation now.",
            "Do not delay urgent transport to ER.",
        ],
    },
    "urgent": {
        "symptoms": [
            "high fever with cough",
            "painful urination",
            "persistent vomiting",
            "asthma symptoms not improving",
            "ankle swelling after fall",
            "eye pain with redness",
            "migraine with nausea",
            "moderate dehydration",
            "ear pain with fever",
            "worsening abdominal pain",
        ],
        "duration": ["for one day", "for two days", "since last night", "today"],
        "severity": ["moderate", "worsening", "persistent", "concerning"],
        "context": ["affecting daily activity", "despite home care", "with poor appetite", "with fatigue"],
        "doctor": [
            "Visit urgent care the same day.",
            "Needs same-day clinical review.",
            "Please attend urgent clinic within 24 hours.",
            "Urgent outpatient assessment is recommended.",
        ],
    },
    "routine": {
        "symptoms": [
            "mild sore throat",
            "seasonal allergies",
            "stable chronic back pain",
            "medicine refill request",
            "annual diabetes follow-up",
            "sleep difficulty without alarm signs",
            "skin care concerns",
            "nutrition counseling request",
            "stable thyroid follow-up",
            "occasional knee discomfort",
        ],
        "duration": ["for a week", "for months", "intermittently", "for a few days"],
        "severity": ["mild", "stable", "low-grade", "unchanged"],
        "context": ["without red flags", "manageable at home", "no acute distress", "non-urgent"],
        "doctor": [
            "Schedule a routine clinic appointment.",
            "Routine follow-up is appropriate.",
            "Book a non-urgent outpatient visit.",
            "Continue home care and review routinely.",
        ],
    },
}


def _augment_dialogue(text: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    replacements = {
        r"\bsevere\b": ["intense", "strong", "serious"],
        r"\bmild\b": ["light", "minor", "low-grade"],
        r"\burgent\b": ["prompt", "immediate", "same-day"],
        r"\broutine\b": ["regular", "scheduled", "standard"],
        r"\bemergency\b": ["ER", "emergency", "acute"],
        r"\btoday\b": ["today", "right now", "this morning"],
    }
    out = text
    for pattern, options in replacements.items():
        if re.search(pattern, out, flags=re.IGNORECASE):
            out = re.sub(pattern, rng.choice(options), out, count=1, flags=re.IGNORECASE)
    return out


def _generate_aspect_dialogues(label: str, count: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    bucket = ASPECTS[label]
    rows = []
    for _ in range(count):
        symptom = rng.choice(bucket["symptoms"])
        duration = rng.choice(bucket["duration"])
        severity = rng.choice(bucket["severity"])
        context = rng.choice(bucket["context"])
        doctor = rng.choice(bucket["doctor"])
        text = (
            f"Patient: I have {severity} {symptom} {duration}, {context}. "
            f"Doctor: {doctor}"
        )
        rows.append({"dialogue": text, "label": label})
    return pd.DataFrame(rows)


def build_training_dataframe(df: pd.DataFrame, target_size: int, random_state: int, aspect_ratio: float) -> pd.DataFrame:
    if len(df) >= target_size:
        return df.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    grouped = {label: grp.reset_index(drop=True) for label, grp in df.groupby("label")}
    labels = sorted(grouped.keys())
    per_class = target_size // len(labels)
    remainder = target_size % len(labels)

    parts = []
    for i, label in enumerate(labels):
        needed = per_class + (1 if i < remainder else 0)
        base = grouped[label]
        rng = np.random.default_rng(random_state + i)
        n_aspect = int(needed * aspect_ratio)
        n_base = max(needed - n_aspect, 0)

        rows = []
        for j in range(n_base):
            src = base.iloc[j % len(base)]
            row = src.copy()
            row["dialogue"] = _augment_dialogue(str(src["dialogue"]), seed=random_state + i * 10000 + j)
            rows.append(row)
        class_df = pd.DataFrame(rows, columns=["dialogue", "label"])
        if n_aspect > 0:
            aspect_df = _generate_aspect_dialogues(label, n_aspect, random_state + i * 50000)
            class_df = pd.concat([class_df, aspect_df], ignore_index=True)
        class_df = class_df.sample(frac=1.0, random_state=int(rng.integers(0, 10_000_000)))
        parts.append(class_df)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["dialogue", "label"]).reset_index(drop=True)
    if len(out) < target_size:
        extra_needed = target_size - len(out)
        extra_parts = []
        per_label_extra = extra_needed // len(labels)
        rem_extra = extra_needed % len(labels)
        for i, label in enumerate(labels):
            n = per_label_extra + (1 if i < rem_extra else 0)
            extra_parts.append(_generate_aspect_dialogues(label, n, random_state + 90000 + i))
        out = pd.concat([out, *extra_parts], ignore_index=True)
    return out.sample(n=target_size, random_state=random_state).reset_index(drop=True)


def main() -> None:
    parser = ArgumentParser(description="Train triage classifier with optional dataset expansion.")
    parser.add_argument("--target-size", type=int, default=6000, help="Training dataset size after augmentation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=0.45,
        help="Fraction of generated data coming from aspect templates (0.0 to 0.9).",
    )
    parser.add_argument(
        "--save-augmented-csv",
        action="store_true",
        help="Save the expanded dataset to data/clinical_dialogues_1000.csv.",
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(DATA_PATH)
    if not {"dialogue", "label"}.issubset(df_raw.columns):
        raise ValueError("CSV must contain 'dialogue' and 'label' columns.")
    if args.target_size < 300:
        raise ValueError("target-size must be at least 300 for stable class stratification.")
    if not (0.0 <= args.aspect_ratio <= 0.9):
        raise ValueError("aspect-ratio must be between 0.0 and 0.9.")

    X_raw = df_raw["dialogue"].astype(str).values
    y_raw = df_raw["label"].astype(str).values
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=0.25,
        random_state=args.random_state,
        stratify=y_raw,
    )
    df_train_raw = pd.DataFrame({"dialogue": X_train_raw, "label": y_train_raw})
    df_test_raw = pd.DataFrame({"dialogue": X_test_raw, "label": y_test_raw})

    df = build_training_dataframe(
        df_train_raw,
        target_size=args.target_size,
        random_state=args.random_state,
        aspect_ratio=args.aspect_ratio,
    )
    if args.save_augmented_csv:
        out_csv = ROOT / "data" / f"clinical_dialogues_{args.target_size}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved expanded dataset: {out_csv}")

    X = df["dialogue"].astype(str).values
    y_text = df["label"].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    y_test = encoder.transform(df_test_raw["label"].astype(str).values)
    X_test = df_test_raw["dialogue"].astype(str).values

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    best_pipeline = search.best_estimator_
    preds = best_pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "classification_report": classification_report(
            y_test,
            preds,
            target_names=encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        ),
        "macro_precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "labels": encoder.classes_.tolist(),
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(search.best_score_),
        "eval_set_size": len(X_test),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "pipeline.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    df.drop_duplicates(subset=["dialogue", "label"]).to_csv(MODELS_DIR / "reference_dialogues.csv", index=False)

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Training set size used: {len(df)}")
    print(f"Holdout size used: {len(X_test)}")
    print(f"Best params: {metrics['best_params']}")
    print(f"Best CV F1-macro: {metrics['best_cv_f1_macro']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved model artifacts to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
