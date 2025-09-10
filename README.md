# Fake_news_detection


# 📰 Fake News Detection App

A Streamlit web app that detects fake news using multiple models:
- Passive Aggressive Classifier
- Multinomial Naive Bayes
- BERT (`omykhailiv/bert-fake-news-recognition`)
- FLAN-T5 (`google/flan-t5-base`)

Final verdict is decided by **majority voting**.

## 🚀 Features
- Enter text OR paste a URL → automatic scraping.
- Runs 4 models internally, gives **one final verdict**.
- Color-coded results (🟢 REAL, 🔴 FAKE).

## 🛠️ Installation
```bash
git clone <repo-link>
cd FakeNewsApp
pip install -r requirements.txt
