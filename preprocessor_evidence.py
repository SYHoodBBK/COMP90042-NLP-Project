import json
import re
import string
import unicodedata
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Global variables for worker processes, will be initialized by init_worker
nlp_worker = None
stop_words_worker = None


def init_worker():
    """Initializer for each worker process in the pool."""
    global nlp_worker, stop_words_worker
    import spacy
    import nltk
    from nltk.corpus import stopwords

    # Ensure NLTK stopwords are downloaded (quietly) if not already present
    # This will be called once per worker process
    nltk.download('stopwords', quiet=True)
    stop_words_worker = set(stopwords.words("english"))
    # Load the spaCy model once per worker process
    nlp_worker = spacy.load("en_core_web_sm")


def normalize_unicode_characters(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def clean_punctuation(text: str) -> str:
    text = re.sub(r'(?<!\d),(?!\d)', ' ', text)
    # text = re.sub(r'[\"\'\(\)\[\]\{\}:;!?@#“”’‘]', ' ', text)
    punctuation_except_comma = string.punctuation.replace(",", "")
    text = re.sub(f"[{re.escape(punctuation_except_comma)}]", " ", text)
    return text


# Define helper functions at the top level for pickling
def create_term_info():
    return {"freq": 0}


def create_doc_map():
    return defaultdict(create_term_info)


def process_evidence_item(item):
    # Imports like spacy, nltk, stopwords are moved to init_worker
    # nltk.download and spacy.load are also moved to init_worker

    def preprocess_text(text: str,
                        normalize_unicode: bool = True,
                        clean_punctuation_flag: bool = True,
                        preserve_numbers: bool = True) -> list:
        if normalize_unicode:
            text = normalize_unicode_characters(text)
        if clean_punctuation_flag:
            text = clean_punctuation(text)
        text = text.strip().lower()
        # Use the worker-specific nlp model
        doc = nlp_worker(text)

        processed = []
        for token in doc:
            if token.is_punct:
                continue
            if not preserve_numbers and token.like_num and not token.is_alpha:
                continue

            lemma = token.lemma_.strip().lower()
            # Use the worker-specific stopwords
            if lemma and lemma not in stop_words_worker:
                processed.append(lemma)

        return processed

    ev_id, text = item
    # Use the top-level helper functions here
    local_index = defaultdict(create_doc_map)
    tokens = preprocess_text(text)

    for token in tokens:
        local_index[token][ev_id]["freq"] += 1


    return {
        "local_index": local_index,
        "processed_text": {ev_id: tokens}
    }


def build_inverted_index_parallel(evidences: dict, num_workers: int = None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_evidence_item, evidences.items()),
                            total=len(evidences), desc="Processing evidence"))

    inverted_index = defaultdict(lambda: defaultdict(lambda: {"freq": 0}))
    processed_texts = {}

    for result in tqdm(results, desc="Merging indexes"):
        local_index = result["local_index"]
        local_processed = result["processed_text"]

        for term, doc_map in local_index.items():
            for doc_id, info in doc_map.items():
                inverted_index[term][doc_id]["freq"] += info["freq"]

        processed_texts.update(local_processed)

    return inverted_index, processed_texts


def main():
    evidence_path = 'data/evidence.json'
    index_output_path = 'ProcessedData/inverted_index.json'
    processed_output_path = 'ProcessedData/processed_evidences.json'

    # Step 1: Load evidences
    with open(evidence_path, 'r', encoding='utf-8') as f:
        evidences = json.load(f)

    print(f"Loaded {len(evidences)} evidences.")

    # Step 2: Build index & collect processed tokens
    start = time.time()
    index, processed = build_inverted_index_parallel(evidences)
    end = time.time()

    print(f"Inverted index built in {end - start:.2f} seconds.")

    # Step 3: Save index
    index_dict = {term: dict(doc_map) for term, doc_map in index.items()}
    with open(index_output_path, "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2)
    print(f"Inverted index saved to {index_output_path}.")

    # Step 4: Save processed evidence tokens
    with open(processed_output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)
    print(f"Processed token sequences saved to {processed_output_path}.")


if __name__ == "__main__":
    main()
