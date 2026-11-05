
import re
import string
import numpy as np
import joblib
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

BASE_DIR = Path(__file__).resolve().parent

loaded_model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
loaded_tfidf = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

email_words = {
    "please", "hi", "hello", "dear", "thanks",
    "thank", "regards", "kind", "best", "pm"
}


def preprocess_ticket(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in email_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def assign_priority(text):
    text = text.lower()

    high_keywords = [
        "critical", "urgent", "system down",
        "security breach", "unable to work",
        "business stopped", "data loss",
        "server failure", "complete outage"
    ]

    medium_keywords = [
        "error", "not working", "access issue",
        "malfunction", "problem", "damaged",
        "broken", "replacement", "password",
        "cannot access"
    ]

    if any(keyword in text for keyword in high_keywords):
        return "High"
    elif any(keyword in text for keyword in medium_keywords):
        return "Medium"
    else:
        return "Low"


def classify_support_ticket(text):
    cleaned_text = preprocess_ticket(text)
    vector = loaded_tfidf.transform([cleaned_text])

    probabilities = loaded_model.predict_proba(vector)[0]
    classes = loaded_model.classes_

    top_indices = np.argsort(probabilities)[::-1][:3]

    category = classes[top_indices[0]]
    confidence = probabilities[top_indices[0]] * 100
    priority = assign_priority(text)

    status = (
        "Needs human review"
        if confidence < 70 or priority == "High"
        else "Prediction accepted"
    )

    return {
        "Category": category,
        "Confidence": round(confidence, 2),
        "Priority": priority,
        "Status": status,
        "Alternative Categories": {
            classes[i]: float(round(probabilities[i] * 100, 2))
            for i in top_indices
        }
    }
