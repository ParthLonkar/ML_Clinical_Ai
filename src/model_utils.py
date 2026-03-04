import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "models" / "pipeline.joblib"
ENCODER_PATH = ROOT / "models" / "label_encoder.joblib"
REFERENCE_PATH = ROOT / "models" / "reference_dialogues.csv"
load_dotenv(ROOT / ".env")

SAFETY_RULES = {
    "emergency": [
        "chest pain",
        "shortness of breath",
        "fainted",
        "slurred speech",
        "seizure",
        "vomiting blood",
        "blue lips",
    ]
}

MEDICAL_HINTS = {
    "pain",
    "fever",
    "cough",
    "breath",
    "vomit",
    "blood",
    "headache",
    "dizzy",
    "faint",
    "chest",
    "rash",
    "swelling",
    "weakness",
    "diarrhea",
    "urination",
    "seizure",
    "allergy",
    "nausea",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "at",
    "by",
    "from",
    "this",
    "that",
    "it",
    "i",
    "have",
    "has",
    "had",
    "my",
    "patient",
    "doctor",
    "since",
    "today",
    "now",
}


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def load_artifacts():
    if not PIPELINE_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Missing model files. Run `python src/train.py` first.")
    return joblib.load(PIPELINE_PATH), joblib.load(ENCODER_PATH)


def load_reference_dialogues() -> pd.DataFrame:
    if REFERENCE_PATH.exists():
        df = pd.read_csv(REFERENCE_PATH)
        if {"dialogue", "label"}.issubset(df.columns):
            return df
    fallback = ROOT / "data" / "clinical_dialogues.csv"
    return pd.read_csv(fallback)


def apply_safety_overrides(text: str, predicted_label: str) -> Tuple[str, str]:
    lower_text = text.lower()
    for keyword in SAFETY_RULES["emergency"]:
        if keyword in lower_text and predicted_label != "emergency":
            return "emergency", f"Safety override: keyword '{keyword}' detected."
    return predicted_label, ""


def explain_text(pipeline, class_index: int, text: str, top_k: int = 8) -> List[Dict[str, float]]:
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    features = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[class_index]

    contributions = features.multiply(coefs).toarray()[0]
    nonzero = np.where(contributions > 0)[0]
    if nonzero.size == 0:
        return []

    ranked = nonzero[np.argsort(contributions[nonzero])[::-1]][:top_k]
    return [
        {"token": str(feature_names[i]), "contribution": float(contributions[i])}
        for i in ranked
    ]


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _token_overlap_score(query_tokens: set, doc_tokens: set) -> Tuple[float, List[str]]:
    if not query_tokens or not doc_tokens:
        return 0.0, []
    matched = sorted(query_tokens.intersection(doc_tokens))
    union = len(query_tokens.union(doc_tokens))
    return (len(matched) / union if union else 0.0), matched


def _retrieve_similar_cases(pipeline, text: str, reference_df: pd.DataFrame, top_n: int = 3):
    vectorizer = pipeline.named_steps["tfidf"]
    ref_texts = reference_df["dialogue"].astype(str).tolist()
    ref_labels = reference_df["label"].astype(str).tolist()
    ref_matrix = vectorizer.transform(ref_texts)
    query = vectorizer.transform([text])
    vector_sims = cosine_similarity(query, ref_matrix)[0]

    query_tokens = set(_tokenize(text))
    token_scores = []
    token_matches = {}
    for i, doc in enumerate(ref_texts):
        score, matched = _token_overlap_score(query_tokens, set(_tokenize(doc)))
        token_scores.append(score)
        token_matches[i] = matched
    token_scores = np.array(token_scores)

    combined_scores = (0.70 * vector_sims) + (0.30 * token_scores)
    sorted_idx = np.argsort(combined_scores)[::-1]
    top_idx = []
    seen_dialogues = set()
    for i in sorted_idx:
        dialogue = ref_texts[i]
        if dialogue in seen_dialogues:
            continue
        seen_dialogues.add(dialogue)
        top_idx.append(i)
        if len(top_idx) >= top_n:
            break

    similar_cases = []
    for i in top_idx:
        similar_cases.append(
            {
                "dialogue": ref_texts[i],
                "label": ref_labels[i],
                "similarity": float(combined_scores[i]),
                "vector_similarity": float(vector_sims[i]),
                "token_overlap": float(token_scores[i]),
                "matched_tokens": token_matches[i],
            }
        )

    support = {}
    for i in top_idx:
        label = ref_labels[i]
        support[label] = support.get(label, 0.0) + float(max(combined_scores[i], 0.0))

    total = sum(support.values())
    if total > 0:
        support = {k: v / total for k, v in support.items()}

    token_reading = {
        "input_tokens": sorted(query_tokens),
        "matched_tokens": sorted({tok for i in top_idx for tok in token_matches[i]}),
    }
    token_reading["match_coverage"] = (
        len(token_reading["matched_tokens"]) / len(token_reading["input_tokens"])
        if token_reading["input_tokens"]
        else 0.0
    )
    return similar_cases, support, token_reading


def _extract_symptoms(text: str) -> List[str]:
    tokens = [w.strip(".,!?;:").lower() for w in text.split()]
    symptom_hits = sorted({t for t in tokens if any(h in t for h in MEDICAL_HINTS)})
    return symptom_hits[:8]


def _is_clinical_text(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in MEDICAL_HINTS)


def _build_suggestion(label: str, text: str, confidence: float) -> str:
    symptoms = _extract_symptoms(text)
    symptom_text = ", ".join(symptoms) if symptoms else "reported symptoms"

    if not _is_clinical_text(text):
        return (
            "Input seems non-clinical. Share concrete symptoms, duration, age, and severity "
            "to receive a useful triage suggestion."
        )
    if label == "emergency":
        return (
            f"Potential high-risk signs detected ({symptom_text}). Seek emergency care now, "
            "avoid self-medication, and do not delay transport."
        )
    if label == "urgent":
        base = (
            f"Likely same-day assessment needed for {symptom_text}. Visit urgent care within 24 hours "
            "and monitor for worsening breathing, bleeding, confusion, or severe pain."
        )
        if confidence < 0.50:
            return base + " Confidence is moderate; if symptoms escalate, treat as emergency."
        return base
    return (
        f"Current pattern suggests routine follow-up for {symptom_text}. "
        "Book a clinic visit, track symptoms for 48-72 hours, and escalate to urgent care if worse."
    )


def _build_gemini_suggestion(
    text: str,
    label: str,
    confidence: float,
    similar_cases: List[Dict],
    fallback_suggestion: str,
) -> Tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_gemini = _env_flag("USE_GEMINI", "false")
    if not use_gemini or not api_key:
        return fallback_suggestion, "local"

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}

    case_lines = []
    for c in similar_cases:
        case_lines.append(f"- {c['label']} (sim={c['similarity']:.2f}): {c['dialogue']}")
    cases_block = "\n".join(case_lines) if case_lines else "- none"

    prompt = (
        "You are a clinical triage assistant for educational prototype use.\n"
        "Given the input dialogue and model output, write a concise, practical suggestion.\n"
        "Rules:\n"
        "1) Do not provide diagnosis.\n"
        "2) Use clear next-step actions.\n"
        "3) Mention emergency escalation signs.\n"
        "4) Keep 2-4 short sentences.\n\n"
        f"Input dialogue: {text}\n"
        f"Model label: {label}\n"
        f"Model confidence: {confidence:.3f}\n"
        f"Similar cases:\n{cases_block}\n"
    )

    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, headers=headers, params={"key": api_key}, json=body, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return fallback_suggestion, "local"
        text_out = (
            candidates[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if not text_out:
            return fallback_suggestion, "local"
        return text_out, "gemini"
    except Exception:
        return fallback_suggestion, "local"


def predict_with_explanation(text: str, top_k: int = 8) -> Dict:
    pipeline, encoder = load_artifacts()
    reference_df = load_reference_dialogues()

    model_probs = pipeline.predict_proba([text])[0]
    labels = encoder.classes_.tolist()
    class_probs = {str(label): float(prob) for label, prob in zip(labels, model_probs.tolist())}

    similar_cases, retrieval_support, token_reading = _retrieve_similar_cases(
        pipeline, text, reference_df, top_n=3
    )
    merged_probs = {}
    for label in labels:
        ml_prob = class_probs.get(str(label), 0.0)
        sim_prob = retrieval_support.get(str(label), 0.0)
        merged_probs[str(label)] = (0.75 * ml_prob) + (0.25 * sim_prob)

    denom = sum(merged_probs.values()) or 1.0
    merged_probs = {k: v / denom for k, v in merged_probs.items()}
    predicted_label = max(merged_probs, key=merged_probs.get)
    confidence = float(merged_probs[predicted_label])

    final_label, rule_note = apply_safety_overrides(text, predicted_label)
    class_index = int(np.where(encoder.classes_ == final_label)[0][0])
    explanation = explain_text(pipeline, class_index, text, top_k=top_k)
    fallback_suggestion = _build_suggestion(final_label, text, confidence)
    suggestion, suggestion_source = _build_gemini_suggestion(
        text=text,
        label=final_label,
        confidence=confidence,
        similar_cases=similar_cases,
        fallback_suggestion=fallback_suggestion,
    )
    uncertainty = "low" if confidence >= 0.65 else "medium" if confidence >= 0.45 else "high"

    return {
        "label": final_label,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "rule_note": rule_note,
        "suggestion": suggestion,
        "suggestion_source": suggestion_source,
        "top_evidence": explanation,
        "class_probabilities": merged_probs,
        "similar_cases": similar_cases,
        "token_reading": token_reading,
    }
