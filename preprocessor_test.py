import json
import re
import string
import unicodedata
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Global variables for worker processes
nlp_worker = None
stop_words_worker = None

def init_worker():
    """Initializer for each worker process."""
    global nlp_worker, stop_words_worker
    import spacy
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords', quiet=True)
    stop_words_worker = set(stopwords.words("english"))
    nlp_worker = spacy.load("en_core_web_sm")

def normalize_unicode_characters(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

def clean_punctuation(text: str) -> str:
    punctuation_except_comma = string.punctuation.replace(",", "")
    return re.sub(f"[{re.escape(punctuation_except_comma)}]", " ", text)

def preprocess_text(text: str,
                    normalize_unicode: bool = True,
                    clean_punctuation_flag: bool = True,
                    preserve_numbers: bool = True) -> list:
    if normalize_unicode:
        text = normalize_unicode_characters(text)
    if clean_punctuation_flag:
        text = clean_punctuation(text)
    text = text.strip().lower()
    doc = nlp_worker(text)

    processed = []
    for token in doc:
        if token.is_punct:
            continue
        if not preserve_numbers and token.like_num and not token.is_alpha:
            continue
        lemma = token.lemma_.strip().lower()
        if lemma and lemma not in stop_words_worker:
            processed.append(lemma)

    return processed

def process_claim_item(item):
    claim_id, claim_data = item
    tokens = preprocess_text(claim_data["claim_text"])
    return claim_id, {
        "claim_text": claim_data["claim_text"],
        "tokens": tokens
    }

def process_claims_parallel(claims: dict, num_workers: int = None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_claim_item, claims.items()),
                            total=len(claims), desc="Processing test claims"))

    processed_claims = {cid: processed for cid, processed in results}
    return processed_claims

def main():
    input_path = 'data/test-claims-unlabelled.json'
    output_path = 'data/processed_test_claims.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        claims = json.load(f)

    print(f"Loaded {len(claims)} test claims.")

    start = time.time()
    processed = process_claims_parallel(claims)
    end = time.time()

    print(f"Processed in {end - start:.2f} seconds.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2)

    print(f"Saved to {output_path}.")

if __name__ == "__main__":
    main()
