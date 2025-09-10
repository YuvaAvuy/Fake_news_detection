import streamlit as st
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article

# -------------------------------
# Load ML models
# -------------------------------
model_pa = joblib.load("model_passive_aggressive.pkl")
model_nb = joblib.load("model_multinb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Hugging Face Models
bert_model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
bert_tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

t5_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# -------------------------------
# Scraper
# -------------------------------
def scrape_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title + "\n\n" + article.text
    except:
        return None

# -------------------------------
# Prediction Function
# -------------------------------
def get_final_prediction(text):
    # Passive Aggressive
    vec = vectorizer.transform([text])
    pa_pred = "REAL" if model_pa.predict(vec)[0] == 1 else "FAKE"

    # Naive Bayes
    nb_pred = "REAL" if model_nb.predict(vec)[0] == 1 else "FAKE"

    # BERT
    bert_res = bert_pipeline(text[:512])[0]
    bert_pred = "REAL" if "REAL" in bert_res['label'].upper() else "FAKE"

    # FLAN-T5
    prompt = f"Classify this news as REAL or FAKE:\n\n{text}"
    t5_out = t5_pipeline(prompt, max_length=20)[0]['generated_text'].upper()
    if "REAL" in t5_out and "FAKE" not in t5_out:
        t5_pred = "REAL"
    elif "FAKE" in t5_out:
        t5_pred = "FAKE"
    else:
        t5_pred = "UNSURE"

    # Majority Voting
    votes = [pa_pred, nb_pred, bert_pred, t5_pred]
    final = max(set(votes), key=votes.count)

    # Tie-breaker with BERT
    if votes.count(final) == 2 and "UNSURE" in votes:
        final = bert_pred

    return final

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Detection App (All Models Combined)")

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
        if final_result == "REAL":
            st.success("🟢 FINAL VERDICT: REAL NEWS")
        else:
            st.error("🔴 FINAL VERDICT: FAKE NEWS")
