import streamlit as st
import joblib
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# Load classical ML models
# ==============================
model_pa = joblib.load("model_passive_aggressive.pkl")
model_nb = joblib.load("model_multinb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==============================
# Load Hugging Face models
# ==============================
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

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Scraper
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")

        title = soup.title.string if soup.title else ""

        article_div = soup.find("article")
        if not article_div:
            article_div = soup.find("div", {"class": "articlebodycontent"})
        if not article_div:
            article_div = soup.find("div", {"id": "content-body"})

        if article_div:
            chunks = [
                elem.get_text().strip()
                for elem in article_div.find_all(["p", "li", "div"])
                if len(elem.get_text().split()) > 5
            ]
        else:
            chunks = [
                p.get_text().strip()
                for p in soup.find_all("p")
                if len(p.get_text().split()) > 5
            ]

        text = " ".join(chunks)
        if not text:
            text = soup.get_text()

        return clean_text((title + "\n\n" + text)[:3000])
    except Exception:
        return None

# ==============================
# Trusted sources list
# ==============================
trusted_sources = [
    # Indian mainstream
    "thehindu.com", "timesofindia.com", "hindustantimes.com", "ndtv.com", "indiatoday.in",
    "indianexpress.com", "economictimes.indiatimes.com", "livemint.com", "business-standard.com",
    "deccanherald.com", "telegraphindia.com", "mid-day.com", "dnaindia.com", "scroll.in",
    "firstpost.com", "theprint.in", "news18.com", "oneindia.com", "outlookindia.com",

    # International mainstream
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "aljazeera.com", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "bloomberg.com", "dw.com", "foxnews.com", "cbsnews.com",
    "nbcnews.com", "abcnews.go.com", "sky.com", "france24.com", "rt.com", "sputniknews.com", "npr.org",

    # Indian government
    ".gov.in", "pib.gov.in", "isro.gov.in", "mea.gov.in", "pmindia.gov.in", "presidentofindia.nic.in",
    "mod.gov.in", "mha.gov.in", "rbi.org.in", "sebi.gov.in",

    # International govt + orgs
    ".gov", ".europa.eu", "un.org", "who.int", "nasa.gov", "esa.int",
    "ecb.europa.eu", "imf.org", "worldbank.org",
]

def is_trusted(url: str) -> bool:
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# Prediction logic with weighted voting
# ==============================
def get_final_prediction(text, url=""):
    text = clean_text(text)

    # Trusted source override
    if url and is_trusted(url):
        return "REAL"

    # Classical ML
    vec = vectorizer.transform([text])
    pa_pred = model_pa.predict(vec)[0]
    nb_pred = model_nb.predict(vec)[0]

    # DL models
    bert_res = bert_pipeline(text[:512])[0]['label']
    bert_pred = "REAL" if "REAL" in bert_res.upper() else "FAKE"

    t5_out = t5_pipeline(f"Classify this news as REAL or FAKE:\n\n{text}", max_length=20)[0]['generated_text'].upper()
    if "REAL" in t5_out and "FAKE" not in t5_out:
        t5_pred = "REAL"
    elif "FAKE" in t5_out:
        t5_pred = "FAKE"
    else:
        t5_pred = "UNSURE"

    # Weighted voting
    scores = {"REAL": 0, "FAKE": 0}
    scores["REAL"] += 0.5 if bert_pred == "REAL" else 0
    scores["FAKE"] += 0.5 if bert_pred == "FAKE" else 0
    scores["REAL"] += 0.3 if t5_pred == "REAL" else 0
    scores["FAKE"] += 0.3 if t5_pred == "FAKE" else 0
    scores["REAL"] += 0.1 if pa_pred == 1 else 0
    scores["FAKE"] += 0.1 if pa_pred == 0 else 0
    scores["REAL"] += 0.1 if nb_pred == 1 else 0
    scores["FAKE"] += 0.1 if nb_pred == 0 else 0

    if scores["REAL"] > scores["FAKE"]:
        return "REAL"
    elif scores["FAKE"] > scores["REAL"]:
        return "FAKE"
    else:
        return "UNSURE"

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection App (Trusted + DL/ML Weighted Ensemble)")

choice = st.radio("Choose Input Type", ["Text", "URL"])
user_input = ""
page_url = ""

if choice == "Text":
    user_input = st.text_area("Enter news text/headline")
else:
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=200)
            user_input = scraped
        else:
            st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some news text or URL.")
    else:
        final_result = get_final_prediction(user_input, page_url)

        st.subheader("Final Verdict:")
        if final_result == "REAL":
            st.success("🟢 REAL NEWS")
        elif final_result == "FAKE":
            st.error("🔴 FAKE NEWS")
        else:
            st.warning("⚠️ UNSURE")

        with st.expander("🔎 Debug: Show Extracted Text"):
            st.write(user_input)
