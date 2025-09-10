import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# Load DL Models
# ==============================
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Web Scraping
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article_div = soup.find("article") or soup.find("div", {"class": "articlebodycontent"}) or soup.find("div", {"id": "content-body"})
        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p","li","div"]) if len(elem.get_text().split())>5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split())>5]
        text = " ".join(chunks)
        if not text:
            text = soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except:
        return None

# ==============================
# Trusted Sources (200+)
# ==============================
trusted_sources = [
    # Example: extend to 200+ sources (national, international, government)
    "thehindu.com","timesofindia.com","bbc.com","cnn.com",".gov.in","un.org","nasa.gov",
    "who.int",".gov",".europa.eu","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in",
    # ... add all 200+ sources as in previous full code
]

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# DL Ensemble Prediction
# ==============================
def predict_text_ensemble(text, url=""):
    text = clean_text(text)

    if url and is_trusted(url):
        return "REAL"

    # Load models
    bert_pipeline = load_bert_model()
    roberta_pipeline = load_roberta_model()

    # BERT prediction
    bert_res = bert_pipeline(text[:512])[0]['label']
    bert_pred = "REAL" if "REAL" in bert_res.upper() else "FAKE"

    # RoBERTa Zero-Shot prediction
    roberta_res = roberta_pipeline(text, candidate_labels=["REAL","FAKE"])
    roberta_pred = roberta_res['labels'][0]

    # Weighted Voting
    scores = {"REAL":0, "FAKE":0}
    for p, w in zip([bert_pred, roberta_pred],[0.5,0.5]):
        if p=="REAL": scores["REAL"] += w
        elif p=="FAKE": scores["FAKE"] += w

    return "REAL" if scores["REAL"] > scores["FAKE"] else ("FAKE" if scores["FAKE"] > scores["REAL"] else "UNSURE")

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection App (DL Ensemble)")

input_type = st.radio("Choose Input Type", ["Text", "URL"])

user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text here", height=200)
else:
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=300)
            user_input = scraped
        else:
            st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter valid text or URL.")
    else:
        try:
            final_result = predict_text_ensemble(user_input, page_url)
            st.subheader("Final Verdict:")
            if final_result=="REAL":
                st.success("🟢 REAL NEWS")
            elif final_result=="FAKE":
                st.error("🔴 FAKE NEWS")
            else:
                st.warning("⚠️ UNSURE")

            with st.expander("🔎 Debug: Show Extracted Text"):
                st.write(user_input)
        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")
