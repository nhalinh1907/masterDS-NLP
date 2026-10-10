"""Train TF-IDF + Multinomial Naive Bayes model and save artifacts to disk."""
import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text


def train(data_path: Path, output_dir: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"

    print(f"Loading dataset from: {data_path}")
    t0 = time.time()
    df = pd.read_csv(data_path)

    print("Cleaning text data...")
    X = df["review"].astype(str).map(clean_text)
    y = (df["sentiment"].str.strip().str.lower() == "positive").astype(int)

    print("Splitting train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting TF-IDF Vectorizer (ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    print("Training Multinomial Naive Bayes model (alpha=0.5)...")
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    print("Evaluating on test set...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n==========================================")
    print(f"Training completed in {time.time() - t0:.2f} seconds")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"==========================================")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    artifacts = {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": float(acc),
    }

    joblib.dump(artifacts, model_path)
    print(f"\nSaved model artifact to: {model_path}")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Naive Bayes Model")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "IMDB Dataset.csv",
        help="Path to IMDB Dataset.csv (fallback to input/IMDB Dataset.csv if missing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory to save model artifact",
    )
    args = parser.parse_args()

    data_file = args.data
    if not data_file.exists():
        fallback = PROJECT_ROOT / "input" / "IMDB Dataset.csv"
        if fallback.exists():
            data_file = fallback

    train(data_file, args.output)
