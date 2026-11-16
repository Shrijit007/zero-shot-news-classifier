import streamlit as st
import pandas as pd

from classifier import DEFAULT_LABELS, NewsClassifier

st.set_page_config(page_title="Zero-Shot News Classifier", page_icon="🏷️", layout="centered")
st.title("Zero-Shot News Classifier")
st.caption("Classify a passage against labels supplied at prediction time.")

@st.cache_resource
def get_classifier():
    return NewsClassifier(device=-1)

with st.form("classification_form"):
    passage = st.text_area(
        "News passage",
        value="The team won the final cricket match after a close contest.",
        height=160,
        placeholder="Enter an English passage...",
    )
    labels_text = st.text_input(
        "Candidate labels (comma-separated)",
        value=", ".join(DEFAULT_LABELS),
    )
    mode = st.radio(
        "Scoring mode",
        ["Single-label ranking", "Independent multi-label scores"],
        horizontal=False,
    )
    submitted = st.form_submit_button("Classify", type="primary")

if submitted:
    labels = [item.strip() for item in labels_text.split(",")]
    try:
        with st.spinner("Loading model (first run may download weights) and classifying..."):
            result = get_classifier().classify(
                passage, labels, multi_label=(mode == "Independent multi-label scores")
            )
        st.subheader("Prediction")
        st.write("**Top-ranked category:**", result["labels"][0])
        table = pd.DataFrame({"Label": result["labels"], "Score": result["scores"]})
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index("Label"))
        st.caption(
            f'Model: {result["model"]} · Revision: {result["revision"] or "default"} · '
            f'Inference: {result["inference_seconds"]:.3f}s'
        )
        st.download_button(
            "Download result JSON",
            data=__import__("json").dumps(result, indent=2),
            file_name="classification_result.json",
            mime="application/json",
        )
        if result["multi_label"]:
            st.info("Scores are independent relevance scores; no selection threshold is applied.")
        else:
            st.info("Scores are normalized over the supplied candidates and depend on the candidate set.")
    except Exception as exc:
        st.error(str(exc))
