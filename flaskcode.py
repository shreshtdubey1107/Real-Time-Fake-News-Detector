import json
import numpy as np
from flask import Flask, render_template, request, url_for, redirect
import joblib
from urllib.parse import urlparse
from phi_verifier import verify_with_phi
from scipy.sparse import csr_matrix, hstack
from newspaper import Article
import requests
from sklearn.ensemble import VotingClassifier
from trustedsources import trusted_domains

app = Flask(__name__)

# -------------------------
# Load Models
# -------------------------

lr_model = joblib.load("news_classifier.pkl")
dt_model = joblib.load("decision_tree_news_classifier.pkl")
rf_model = joblib.load("random_forest_news_classifier.pkl")
knn_model = joblib.load("knn_news_classifier.pkl")
ann_model = joblib.load("ann_news_classifier.pkl")
xgb_model = joblib.load("xgboost_news_classifier.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
)


# -------------------------
# Trusted Source Check
# -------------------------

def is_trusted(url):
    try:
        domain = urlparse(url).netloc.lower()
        return any(t in domain for t in trusted_domains)
    except:
        return False


# -------------------------
# Fetch Article
# -------------------------

def fetch_article_text(url):
    resp = requests.get(
        url,
        headers={'User-Agent': USER_AGENT},
        timeout=10
    )

    resp.raise_for_status()

    article = Article(url)
    article.download(input_html=resp.text)
    article.parse()

    return article.title, article.text


# -------------------------
# Ensemble Prediction
# -------------------------
weights = np.array([
    2.0,
    4.5,
    2.0,
    2.0,
    2.0,
    4.5
])
def predict_fake_news(text, url=None):

    X_text = vectorizer.transform([text])

    trust_flag = 1 if url and is_trusted(url) else 0

    X_flag = csr_matrix([[trust_flag]])

    X = hstack([X_text, X_flag])

    probabilities = []

    probabilities.append(lr_model.predict_proba(X)[0][1])

    probabilities.append(dt_model.predict_proba(X)[0][1])

    probabilities.append(rf_model.predict_proba(X)[0][1])

    probabilities.append(knn_model.predict_proba(X)[0][1])

    probabilities.append(ann_model.predict_proba(X)[0][1])

    probabilities.append(xgb_model.predict_proba(X)[0][1])
    probabilities = np.array(probabilities)
    weighted_probability = np.sum(probabilities * weights) / np.sum(weights)

    score = weighted_probability
    phi_score=verify_with_phi(text)
    print(score*100)
    print(phi_score)
    ans=0

    if trust_flag:
        score = min(score + 0.1, 0.94)
    if phi_score is not None:
        ans = round(((score*100)+4*phi_score)/5,2)

    return ans






# -------------------------
# Home
# -------------------------

@app.route('/')
def home():
    return render_template('index.html')


# -------------------------
# Search Page
# -------------------------

@app.route('/search', methods=['GET', 'POST'])
def search():

    if request.method == 'GET':
        return render_template('search.html')

    url = request.form.get('url', '').strip()
    text = request.form.get('text', '').strip()

    if not url and not text:
        return redirect(url_for('home'))

    return render_template(
        'search.html',
        url=url,
        text=text
    )


# -------------------------
# Prediction
# -------------------------

@app.route('/predict', methods=['POST'])
def predict():

    url = request.form.get('url', '').strip()
    text = request.form.get('text', '').strip()

    title = ""
    article_text = ""
    prediction = None

    try:

        if text:

            title = "User Provided Text"
            article_text = text

            prediction = predict_fake_news(text)

        elif url:

            title, article_text = fetch_article_text(url)

            prediction = predict_fake_news(article_text, url)

        else:

            raise ValueError(
                "Please provide either a URL or text input."
            )

    except Exception as e:

        title = "Error"

        article_text = (
            f"Could not process input.\n\nError: {e}"
        )

    return render_template(
        "result.html",
        title=title,
        article=article_text,
        prediction=prediction,
        url=url,
        text_provided=bool(text)
    )


# -------------------------
# Run App
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)