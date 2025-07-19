import streamlit as st
import joblib
import requests

# Load models
model_pa = joblib.load("model_passive_aggressive.pkl")
model_nb = joblib.load("model_multinb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Gemini Function
def ask_gemini(sample, pa_label, nb_label):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": "AIzaSyARNI1P2Lo7XTol9cjWS6p_TY_6oEt1Qaw"
    }
    prompt = f"""You are a fake news expert.
News: "{sample}"

PassiveAggressive says: {"REAL" if pa_label == 1 else "FAKE"}
MultinomialNB says: {"REAL" if nb_label == 1 else "FAKE"}

Using your world knowledge, decide: is this REAL or FAKE news? Just say REAL or FAKE."""
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        result = res.json()
        text = result['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        if "REAL" in text and "FAKE" not in text:
            return "🟢 REAL NEWS"
        elif "FAKE" in text:
            return "🔴 FAKE NEWS"
        else:
            return "⚠️ Gemini unsure"
    except:
        return "⚠️ Gemini failed"

# UI
st.title("📰 Fake News Detection App")
user_input = st.text_area("Enter a news headline or text")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vec = vectorizer.transform([user_input])
        pa_pred = model_pa.predict(vec)[0]
        nb_pred = model_nb.predict(vec)[0]
        result = ask_gemini(user_input, pa_pred, nb_pred)
        st.subheader("Final Verdict:")
        st.success(result)
