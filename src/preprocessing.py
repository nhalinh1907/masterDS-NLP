"""Source modules for Sentiment Analysis text processing and model handling."""
import re
import string

NEGATION_RULES = [
    (r"won't", "will not"),
    (r"can't", "can not"),
    (r"n't", " not"),
    (r"'re", " are"),
    (r"'s", " is"),
    (r"'d", " would"),
    (r"'ll", " will"),
    (r"'ve", " have"),
    (r"'m", " am"),
]


def clean_text(text: str) -> str:
    """Preprocess movie review text:
    - Lowercase
    - Remove URL links
    - Remove HTML tags
    - Expand contractions (preserving 'not' for negative sentiment)
    - Remove digits and punctuation
    - Normalize whitespace
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>+", " ", text)
    for pat, rep in NEGATION_RULES:
        text = re.sub(pat, rep, text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    return re.sub(r"\s+", " ", text).strip()
