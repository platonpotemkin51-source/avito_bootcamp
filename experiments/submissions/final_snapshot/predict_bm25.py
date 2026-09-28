"""Сформировать answer_bm25.csv после выполнения solution_final.ipynb.

Используются сохранённые TF-IDF-индексы и модель. BM25 строится один раз
из исходного корпуса, если готового индекса пока нет.
"""

import joblib
import numpy as np
import pandas as pd
import os

import bm25_experiment as search
from validate_answer import validate


def main():
    """Загружает готовые индексы и формирует проверенный ответ бенчмарка."""
    search.load_notebook_functions({
        "clean_text", "top_positions", "retrieve_candidates", "make_features"
    })

    columns = [
        "item_id", "item_title_raw", "item_description_raw",
        "item_infm_params_text", "item_category_id", "item_location_id",
        "item_rating", "item_rating_reviews_count", "item_price",
        "item_is_phone_hidden", "item_is_message_forbidden",
    ]
    items = pd.read_parquet("data/benchmark_items.parquet", columns=columns)
    queries = pd.read_parquet("data/benchmark_queries.parquet")
    search.item_ids = items["item_id"].to_numpy()
    search.item_locations = items["item_location_id"].to_numpy()
    search.item_categories = items["item_category_id"].to_numpy()
    search.location_positions = items.groupby("item_location_id").indices
    search.parameter_tokens = [
        set(search.clean_text(text).split())
        for text in items["item_infm_params_text"]
    ]
    search.item_ratings = pd.to_numeric(
        items["item_rating"], errors="coerce"
    ).to_numpy(dtype=float)
    search.item_reviews = pd.to_numeric(
        items["item_rating_reviews_count"], errors="coerce"
    ).to_numpy(dtype=float)
    search.item_prices = pd.to_numeric(
        items["item_price"], errors="coerce"
    ).to_numpy(dtype=float)
    search.phone_hidden = items["item_is_phone_hidden"].fillna(False).to_numpy(dtype=float)
    search.message_forbidden = items["item_is_message_forbidden"].fillna(False).to_numpy(dtype=float)

    tfidf_path = ("outputs/final/indexes.joblib"
                  if os.path.exists("outputs/final/indexes.joblib")
                  else "outputs/v2/indexes.joblib")
    tfidf = joblib.load(tfidf_path)
    search.word_vectorizer = tfidf["word_vectorizer"]
    search.char_vectorizer = tfidf["char_vectorizer"]
    search.word_vectors = tfidf["word_vectors"]
    search.char_vectors = tfidf["char_vectors"]
    bm25_path = "outputs/bm25/index.joblib"
    if os.path.exists(bm25_path):
        bm25 = joblib.load(bm25_path)
        search.bm25_vectorizer = bm25["vectorizer"]
        search.bm25_vectors = bm25["matrix"]
    else:
        os.makedirs("outputs/bm25", exist_ok=True)
        titles = items["item_title_raw"].map(search.clean_text)
        descriptions = items["item_description_raw"].str.slice(0, 1500).map(
            search.clean_text
        )
        parameters = items["item_infm_params_text"].map(search.clean_text)
        corpus_text = titles + " " + titles + " " + parameters + " " + descriptions
        search.bm25_vectorizer, search.bm25_vectors = search.build_bm25(
            corpus_text, search.word_vectorizer.vocabulary_
        )
        joblib.dump({"vectorizer": search.bm25_vectorizer,
                     "matrix": search.bm25_vectors,
                     "k1": search.K1, "b": search.B}, bm25_path)

    model_path = ("outputs/final/selector.joblib"
                  if os.path.exists("outputs/final/selector.joblib")
                  else "outputs/v3/model_10000.joblib")
    model = joblib.load(model_path)["model"]
    print("Индексы и модель загружены.", flush=True)

    original_records = search.retrieve_candidates(queries)
    _, _, extended_records = search.retrieve_bm25(queries, original_records)
    predictions = search.predict_with(model, queries, extended_records)
    answers = [" ".join(search.item_ids[positions]) for positions in predictions]
    answer = pd.DataFrame({"query_id": queries["query_id"], "answer": answers})
    answer.to_csv("answer_bm25.csv", index=False, encoding="utf-8")
    validate("answer_bm25.csv")
    print("Ответ BM25 готов.", flush=True)


if __name__ == "__main__":
    main()
