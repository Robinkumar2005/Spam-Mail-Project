import streamlit as st
import requests
import pandas as pd
import re

# ==========================
# API Configuration
# ==========================
import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
)

API_URL = f"{API_BASE_URL}/batch_predict"
HEALTH_URL = f"{API_BASE_URL}/health"

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="📧 Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# ==========================
# Label Mapping
# ==========================

label_map = {
    "ham": {
        "label": "Ham (Not Spam)",
        "emoji": "✅",
        "color": "#2E7D32"
    },
    "spam": {
        "label": "Spam",
        "emoji": "🚫",
        "color": "#C62828"
    }
}

# ==========================
# Spam Word Highlighter
# ==========================

spammy_words = {
    "free": "Common spam trigger word",
    "win": "Common spam trigger word",
    "winner": "Common spam trigger word",
    "prize": "Common spam trigger word",
    "gift": "Common spam trigger word",
    "cash": "Common spam trigger word",
    "offer": "Common spam trigger word",
    "urgent": "Common spam trigger word",
    "claim": "Common spam trigger word",
    "buy now": "Common spam trigger phrase"
}


def highlight_spammy_words(text):
    highlighted_text = text

    for word, tooltip in spammy_words.items():
        highlighted_text = re.sub(
            f"\\b({re.escape(word)})\\b",
            f'<span style="background-color:yellow;color:black;font-weight:bold;" title="{tooltip}">\\1</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )

    return highlighted_text


# ==========================
# Health Check
# ==========================

def check_api_status():
    try:
        response = requests.get(HEALTH_URL, timeout=5)

        if response.status_code == 200:
            st.sidebar.success("🟢 API Online")
            return True
        else:
            st.sidebar.error("🔴 API Offline")
            return False

    except Exception:
        st.sidebar.error("🔴 API Offline")
        return False


# ==========================
# Prediction Function
# ==========================

def predict_spam_batch(emails):

    try:
        payload = {
            "texts": emails
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return data["results"]

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


# ==========================
# Header
# ==========================

st.title("📧 AI Email Spam Detector")

st.markdown("""
Detect whether emails are **Spam** or **Ham** using your deployed FastAPI machine learning model.

### Features
✅ Batch Prediction  
✅ FastAPI Integration  
✅ Spam Word Highlighting  
✅ CSV Export  
✅ Modern UI  
""")

# ==========================
# Sidebar
# ==========================

st.sidebar.header("System Status")
check_api_status()

st.sidebar.header("Tips")

st.sidebar.info("""
- Start emails with Subject:
- Paste multiple emails
- One email per Subject:
- Avoid empty emails
""")

# ==========================
# Input Area
# ==========================

user_input = st.text_area(
    "Paste Email(s)",
    height=250,
    placeholder="""
Subject: Free Gift Card

Congratulations!
You have won a free gift card.

Subject: Team Meeting

Hi Team,
Let's meet tomorrow at 10 AM.
"""
)

# ==========================
# Predict Button
# ==========================

if st.button("🚀 Predict Emails", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter email content.")
        st.stop()

    emails = [
        email.strip()
        for email in re.split(r'(?=Subject:)', user_input)
        if email.strip()
    ]

    with st.spinner("Analyzing emails..."):

        predictions = predict_spam_batch(emails)

    if predictions:

        st.success(f"Processed {len(emails)} email(s)")

        all_results = []

        for idx, (email, prediction) in enumerate(
            zip(emails, predictions),
            start=1
        ):

            pred = prediction["prediction"].lower()

            label = label_map[pred]["label"]
            emoji = label_map[pred]["emoji"]
            color = label_map[pred]["color"]

            highlighted_email = highlight_spammy_words(email)

            all_results.append({
                "Email": f"Email {idx}",
                "Prediction": label
            })

            with st.expander(
                f"Email {idx} → {emoji} {label}",
                expanded=False
            ):

                st.markdown(
                    f"""
                    <div style="
                        border-left: 8px solid {color};
                        padding: 15px;
                        border-radius: 10px;
                        background-color: #f5f5f5;
                        max-height: 300px;
                        overflow-y: auto;
                    ">
                    {highlighted_email}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    ### Prediction
                    **{emoji} {label}**
                    """
                )

        # ==========================
        # Download CSV
        # ==========================

        st.subheader("📥 Export Results")

        df = pd.DataFrame(all_results)

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="spam_predictions.csv",
            mime="text/csv"
        )

        st.dataframe(df, use_container_width=True)

# ==========================
# Footer
# ==========================

st.markdown("---")
st.caption("Powered by FastAPI + SVM + TF-IDF + Streamlit")
