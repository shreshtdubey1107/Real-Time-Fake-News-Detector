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