import pandas as pd
import streamlit as st

from src.model_utils import predict_with_explanation


st.set_page_config(page_title="Clinical Triage XAI", page_icon=":hospital:", layout="centered")
st.title("Explainable Clinical Decision System")
st.caption("Educational prototype: not medical advice.")

default_text = "Patient reports severe chest pain and shortness of breath."
text = st.text_area("Clinical dialogue", value=default_text, height=180)
top_k = st.slider("Top evidence tokens", min_value=3, max_value=15, value=8, step=1)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Enter dialogue text first.")
    else:
        try:
            result = predict_with_explanation(text, top_k=top_k)
            st.subheader("Prediction")
            st.write(f"**Label:** {result['label'].title()}")
            st.write(f"**Confidence:** {result['confidence'] * 100:.1f}%")
            st.write(f"**Uncertainty:** {result['uncertainty']}")
            if result["rule_note"]:
                st.info(result["rule_note"])
            st.subheader("AI Suggested Action")
            st.caption(f"Suggestion source: {result['suggestion_source']}")
            st.write(result["suggestion"])

            st.subheader("Class Probabilities")
            probs_df = pd.DataFrame(
                [{"class": k, "probability": v} for k, v in result["class_probabilities"].items()]
            ).sort_values("probability", ascending=False)
            probs_df["class"] = probs_df["class"].str.title()
            probs_df["probability"] = (probs_df["probability"] * 100).round(2).astype(str) + "%"
            st.dataframe(probs_df, width="stretch", hide_index=True)

            st.subheader("Top Evidence Tokens")
            if result["top_evidence"]:
                evidence_df = pd.DataFrame(result["top_evidence"])
                evidence_df["contribution"] = evidence_df["contribution"].round(4)
                st.dataframe(evidence_df, width="stretch", hide_index=True)
            else:
                st.write("No strong positive token evidence found.")

            st.subheader("Token Reading")
            st.write(
                f"**Match coverage:** {result['token_reading']['match_coverage'] * 100:.1f}% "
                f"({len(result['token_reading']['matched_tokens'])}/{len(result['token_reading']['input_tokens'])} tokens matched)"
            )
            st.write(
                "**Matched tokens:** "
                + (
                    ", ".join(result["token_reading"]["matched_tokens"])
                    if result["token_reading"]["matched_tokens"]
                    else "none"
                )
            )

            st.subheader("Most Similar Past Cases")
            cases_df = pd.DataFrame(result["similar_cases"])
            cases_df["label"] = cases_df["label"].str.title()
            cases_df["similarity"] = (cases_df["similarity"] * 100).round(1).astype(str) + "%"
            cases_df["vector_similarity"] = (cases_df["vector_similarity"] * 100).round(1).astype(str) + "%"
            cases_df["token_overlap"] = (cases_df["token_overlap"] * 100).round(1).astype(str) + "%"
            cases_df["matched_tokens"] = cases_df["matched_tokens"].apply(lambda x: ", ".join(x) if x else "none")
            st.dataframe(cases_df, width="stretch", hide_index=True)
        except FileNotFoundError as e:
            st.error(str(e))
            st.code("python src/train.py --target-size 1000")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
