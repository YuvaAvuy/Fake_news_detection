import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# Model Loading Functions (Lazy Loaded)
# ==============================
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_bart_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
    # Indian News
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in",
    "indianexpress.com","livemint.com","business-standard.com","deccanherald.com",
    "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com",
    "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com",
    "cnnnews18.com","economictimes.indiatimes.com","financialexpress.com","siasat.com",
    "newindianexpress.com","tribuneindia.com","asianage.com","bharattimes.com",
    "freepressjournal.in","morningindia.in","abplive.com","newsable.asianetnews.com",
    "indiaglitz.com","thelogicalindian.com","m.timesofindia.com","bharatnews.com",
    "sundayguardianlive.com","telanganatoday.com","hyderabadnews.in","bangaloremirror.indiatimes.com",
    "newsnation.in","thenewsminute.com","newslaundry.com","india.com","deccanchronicle.com",
    "thehansindia.com","punemirror.com","chennailivenews.in","kashmirlife.net","jagran.com",
    "navbharattimes.indiatimes.com","amarujala.com","dainikbhaskar.com","lokmat.com",
    "maharashtratimes.com","eenadu.net","sakal.com","prahaar.in","varthabharati.in",
    "samacharjagat.com","dailyhunt.in","uttarpradesh.org",

    # International News
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com",
    "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com",
    "nbcnews.com","abcnews.go.com","sky.com","france24.com","rt.com","sputniknews.com",
    "npr.org","telegraph.co.uk","thetimes.co.uk","independent.co.uk","globaltimes.cn",
    "china.org.cn","cbc.ca","abc.net.au","smh.com.au","japantimes.co.jp","lemonde.fr",
    "elpais.com","derstandard.at","spiegel.de","tagesschau.de","asiatimes.com",
    "straitstimes.com","thaiworldview.com","thejakartapost.com","thestandard.com.hk",
    "sbs.com.au","hawaiinewsnow.com","theglobeandmail.com","irishnews.com","latimes.com",
    "chicagotribune.com","startribune.com","nydailynews.com","financialtimes.com",
    "forbes.com","thehill.com","vox.com","buzzfeednews.com","huffpost.com","usatoday.com",
    "teleSURenglish.net","euronews.com","al-monitor.com",

    # Indian Government
    ".gov.in","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in","mha.gov.in",
    "rbi.org.in","sebi.gov.in","nic.in","mohfw.gov.in","moef.gov.in","meity.gov.in",
    "railway.gov.in","dgca.gov.in","drdo.gov.in","indianrailways.gov.in","education.gov.in",
    "scienceandtech.gov.in","urbanindia.nic.in","financialservices.gov.in",
    "commerce.gov.in","sportsauthorityofindia.nic.in","agriculture.gov.in","power.gov.in",
    "parliamentofindia.nic.in","taxindia.gov.in","cbic.gov.in","epfindia.gov.in","defence.gov.in",

    # International Government & UN/NGO
    ".gov",".europa.eu","un.org","who.int","nasa.gov","esa.int","imf.org","worldbank.org",
    "fao.org","wto.org","unicef.org","unhcr.org","redcross.org","cdc.gov","nih.gov","usa.gov",
    "canada.ca","gov.uk","australia.gov.au","japan.go.jp","europa.eu","consilium.europa.eu",
    "ec.europa.eu","ecb.europa.eu","unep.org","ilo.org","ohchr.org","unodc.org","unwomen.org",
    "unfpa.org","unesco.org","wmo.int","ifrc.org","nato.int","oecd.org","europarl.europa.eu",
    "unido.org","wfp.org"
]

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# DL Ensemble Prediction
# ==============================
def get_final_prediction(text, url=""):
    text = clean_text(text)
    
    if url and is_trusted(url):
        return "REAL"

    # BERT
    bert_res = bert_pipeline(text[:512])[0]['label']
    bert_pred = "REAL" if "REAL" in bert_res.upper() else "FAKE"

    # BART Zero-Shot
    bart_res = bart_pipeline(text, candidate_labels=["REAL","FAKE"])
    bart_pred = bart_res['labels'][0]

    # Weighted Voting
    scores = {"REAL":0, "FAKE":0}
    for p, w in zip([bert_pred, bart_pred],[0.5,0.5]):
        if p=="REAL": scores["REAL"] += w
        elif p=="FAKE": scores["FAKE"] += w

    return "REAL" if scores["REAL"] > scores["FAKE"] else ("FAKE" if scores["FAKE"] > scores["REAL"] else "UNSURE")

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection App (DL Ensemble)")

page_url = st.text_input("Enter news article URL")
user_input = ""

if page_url:
    scraped = scrape_url(page_url)
    if scraped:
        st.text_area("Extracted Article", scraped, height=300)
        user_input = scraped
    else:
        st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a valid URL.")
    else:
        try:
            # Lazy-load models to prevent memory crash
            bert_pipeline = load_bert_model()
            bart_pipeline = load_bart_model()

            final_result = get_final_prediction(user_input, page_url)
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
