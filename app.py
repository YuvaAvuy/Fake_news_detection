import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# Load classical ML models
# -------------------------------
model_pa = joblib.load("model_passive_aggressive.pkl")
model_nb = joblib.load("model_multinb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------
# Load Hugging Face models (cached for speed)
# -------------------------------
@st.cache_resource
def load_bert():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_t5():
    return pipeline("text2text-generation", model="google/flan-t5-base")

bert_pipeline = load_bert()
t5_pipeline = load_t5()

# -------------------------------
# Improved Scraper function
# -------------------------------
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")

        title = soup.title.string if soup.title else ""

        # Extract only meaningful <p> content
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if len(text.split()) > 5:  # ignore short junk like dates/menus
                paragraphs.append(text)

        text = " ".join(paragraphs)

        # Fallback if no good paragraphs found
        if not text:
            text = soup.get_text()

        return (title + "\n\n" + text)[:3000]  # limit to 3000 chars
    except Exception:
        return None

# -------------------------------
# Prediction function
# -------------------------------
def get_final_prediction(text):
    # BERT Prediction
    bert_res = bert_pipeline(text[:512])[0]
    bert_pred = "REAL" if "REAL" in bert_res['label'].upper() else "FAKE"

    # FLAN-T5 Prediction
    prompt = f"Classify this news as REAL or FAKE:\n\n{text}"
    t5_out = t5_pipeline(prompt, max_length=20)[0]['generated_text'].upper()
    if "REAL" in t5_out and "FAKE" not in t5_out:
        t5_pred = "REAL"
    elif "FAKE" in t5_out:
        t5_pred = "FAKE"
    else:
        t5_pred = "UNSURE"

    # Classical Models
    vec = vectorizer.transform([text])
    pa_pred = "REAL" if model_pa.predict(vec)[0] == 1 else "FAKE"
    nb_pred = "REAL" if model_nb.predict(vec)[0] == 1 else "FAKE"

    # Voting
    votes = [bert_pred, t5_pred, pa_pred, nb_pred]
    final = max(set(votes), key=votes.count)

    # Tie → BERT priority
    if votes.count("REAL") == votes.count("FAKE"):
        final = bert_pred

    return final

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Detection App (DL + ML Combined)")

choice = st.radio("Choose Input Type", ["Text", "URL"])

user_input = ""
if choice == "Text":
    user_input = st.text_area("Enter news text/headline")
else:
    url = st.text_input("Enter news article URL")
    if url:
        scraped = scrape_url(url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=200)
            user_input = scraped
        else:
            st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some news text or URL.")
    else:
        final_result = get_final_prediction(user_input)

        st.subheader("Final Verdict:")
        if final_result == "REAL":
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        # Debug: show what models saw
        with st.expander("🔎 Debug: Show Extracted Text"):
            st.write(user_input)
