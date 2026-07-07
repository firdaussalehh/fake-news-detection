import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="📰",
    layout="wide"
)

if "history" not in st.session_state:
    st.session_state.history = []

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Fake News Matcher",
        "Data Explorer",
        "Evaluation Metrics",
        "Prediction Result Table",
        "About Project"
    ]
)

# HOME
elif menu == "Fake News Matcher":
    st.title("🔍 Fake News Matcher")
    st.write("Choose a sample news article or paste your own news text.")

    sample_news = {
        "Select sample news": "",

        "Real Sample: Government Policy":
        "The government announced a new economic policy today according to official sources and ministry reports.",

        "Real Sample: University Research":
        "A university research team published new findings after conducting a study with verified data and expert review.",

        "Real Sample: Police Statement":
        "Police released an official statement about the investigation and confirmed that further action will be taken.",

        "Fake Sample: Miracle Cure":
        "Breaking shocking news! Doctors are hiding a miracle cure that can heal every disease instantly. Click here before it is deleted.",

        "Fake Sample: Secret Government Claim":
        "You will not believe this secret viral report. The government is hiding the truth from everyone and only this website knows it.",

        "Fake Sample: Guaranteed Money":
        "This secret method is guaranteed to make anyone rich overnight. Thousands are using it and banks are trying to stop it."
    }

    selected_sample = st.selectbox("Choose sample news:", list(sample_news.keys()))

    news_text = st.text_area(
        "Or paste your own news article here:",
        value=sample_news[selected_sample],
        height=220
    )

    if st.button("Detect News"):
        if news_text.strip() == "":
            st.warning("Please choose a sample or enter news text.")
        else:
            cleaned = clean_text(news_text)
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]
            probability = model.predict_proba(transformed)[0]
            confidence = max(probability) * 100

            result = "Real News" if prediction == 1 else "Fake News"

            st.session_state.history.append({
                "News Text": news_text[:120] + "...",
                "Prediction": result,
                "Confidence (%)": round(confidence, 2)
            })

            if prediction == 1:
                st.success(f"✅ Prediction: {result}")
            else:
                st.error(f"❌ Prediction: {result}")

            st.metric("Confidence Score", f"{confidence:.2f}%")

# DATA EXPLORER
elif menu == "Data Explorer":
    st.title("📂 Data Explorer")

    st.write("""
    Dataset used: **WELFake Dataset**

    The dataset contains news titles, article text, and labels.
    """)

    data_info = pd.DataFrame({
        "Column": ["title", "text", "label"],
        "Description": [
            "News headline",
            "Full news article",
            "News category label"
        ]
    })

    st.table(data_info)

    label_info = pd.DataFrame({
        "Label": [0, 1],
        "Meaning": ["Fake News", "Real News"]
    })

    st.subheader("Dataset Labels")
    st.table(label_info)

    st.info("Note: The full dataset is used during model training in Google Colab.")

# EVALUATION METRICS
elif menu == "Evaluation Metrics":
    st.title("📊 Evaluation Metrics")

    st.write("Replace these values with your actual results from Google Colab.")

    accuracy = 0.96
    precision = 0.96
    recall = 0.96
    f1 = 0.96

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Precision", f"{precision*100:.2f}%")
    col3.metric("Recall", f"{recall*100:.2f}%")
    col4.metric("F1 Score", f"{f1*100:.2f}%")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [accuracy, precision, recall, f1]
    })

    st.subheader("Evaluation Chart")
    fig, ax = plt.subplots()
    ax.bar(metrics_df["Metric"], metrics_df["Score"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    cm = [[6800, 250],
          [300, 7050]]

    fig2, ax2 = plt.subplots()
    ax2.imshow(cm)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Fake", "Real"])
    ax2.set_yticklabels(["Fake", "Real"])

    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i][j], ha="center", va="center")

    st.pyplot(fig2)

# PREDICTION RESULT TABLE
elif menu == "Prediction Result Table":
    st.title("📋 Prediction Result Table")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Go to Fake News Matcher first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df, use_container_width=True)

        st.subheader("Prediction Summary")

        total = len(history_df)
        fake_count = len(history_df[history_df["Prediction"] == "Fake News"])
        real_count = len(history_df[history_df["Prediction"] == "Real News"])
        avg_confidence = history_df["Confidence (%)"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Predictions", total)
        col2.metric("Fake News", fake_count)
        col3.metric("Real News", real_count)
        col4.metric("Average Confidence", f"{avg_confidence:.2f}%")

        st.subheader("Prediction Distribution")

        chart_data = history_df["Prediction"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(chart_data, labels=chart_data.index, autopct="%1.1f%%")
        st.pyplot(fig)

# ABOUT PROJECT
elif menu == "About Project":
    st.title("ℹ️ About Project")

    st.write("""
    ## Fake News Detection Dashboard Using NLP

    This project is developed to detect fake news using Natural Language Processing and Machine Learning.

    ### Problem Statement
    Fake news spreads quickly online and can mislead people. Therefore, an automated system is useful to help identify whether a news article is likely to be real or fake.

    ### Objectives
    - To preprocess news text using NLP techniques.
    - To classify news articles as Real or Fake.
    - To evaluate the model using accuracy, precision, recall, and F1-score.
    - To display results through an interactive dashboard.

    ### Methodology
    - Dataset: WELFake Dataset
    - Text preprocessing: lowercase conversion, punctuation removal, cleaning
    - Feature extraction: TF-IDF Vectorizer
    - Model: Logistic Regression
    - Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix

    ### Future Improvement
    The system can be improved by using deep learning models such as LSTM, BERT, or transformer-based models.
    """)
