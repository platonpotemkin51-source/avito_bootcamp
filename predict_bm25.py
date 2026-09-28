"""Сформировать финальный answer.csv после выполнения solution.ipynb.

Используются сохранённые TF-IDF-индексы и модель. BM25 строится один раз
из исходного корпуса, если готового индекса пока нет.
"""

import os

import joblib
import pandas as pd

import bm25 as search
from validate_answer import validate


def main():
    """Загружает готовые индексы и формирует проверенный ответ бенчмарка."""
    columns = [
        "item_id", "item_title_raw", "item_description_raw",
        "item_infm_params_text", "item_category_id", "item_location_id",
        "item_rating", "item_rating_reviews_count", "item_price",
        "item_is_phone_hidden", "item_is_message_forbidden",
    ]
    items = pd.read_parquet("data/benchmark_items.parquet", columns=columns)
    queries = pd.read_parquet("data/benchmark_queries.parquet")

    item_ids = items["item_id"].to_numpy()
    location_positions = items.groupby("item_location_id").indices

    item_features = {}
    item_features["locations"] = items["item_location_id"].to_numpy()
    item_features["categories"] = items["item_category_id"].to_numpy()
    item_features["parameter_tokens"] = [
        set(search.clean_text(text).split())
        for text in items["item_infm_params_text"]
    ]

    item_features["ratings"] = pd.to_numeric(
        items["item_rating"], errors="coerce"
    ).to_numpy(dtype=float)
    item_features["reviews"] = pd.to_numeric(
        items["item_rating_reviews_count"], errors="coerce"
    ).to_numpy(dtype=float)
    item_features["prices"] = pd.to_numeric(
        items["item_price"], errors="coerce"
    ).to_numpy(dtype=float)
    item_features["phone_hidden"] = items["item_is_phone_hidden"].fillna(False).to_numpy(dtype=float)
    item_features["message_forbidden"] = items["item_is_message_forbidden"].fillna(False).to_numpy(dtype=float)

    tfidf = joblib.load("outputs/final/indexes.joblib")
    bm25_path = "outputs/bm25/index.joblib"

    if os.path.exists(bm25_path):
        bm25_index = joblib.load(bm25_path)
    else:
        os.makedirs("outputs/bm25", exist_ok=True)
        titles = items["item_title_raw"].map(search.clean_text)
        descriptions = items["item_description_raw"].str.slice(0, 1500).map(
            search.clean_text
        )

        parameters = items["item_infm_params_text"].map(search.clean_text)
        corpus_text = titles + " " + titles + " " + parameters + " " + descriptions
        bm25_vectorizer, bm25_vectors = search.build_bm25(
            corpus_text, tfidf["word_vectorizer"].vocabulary_
        )

        bm25_index = {"vectorizer": bm25_vectorizer, "matrix": bm25_vectors,
                      "k1": search.K1, "b": search.B}
        joblib.dump(bm25_index, bm25_path)

    model = joblib.load("outputs/final/selector.joblib")["model"]
    print("Индексы и модель загружены.", flush=True)

    original_records = search.retrieve_candidates(queries, tfidf, location_positions)
    _, _, extended_records = search.retrieve_bm25(
        queries, original_records, tfidf, bm25_index,
        location_positions, item_features["locations"],
    )
    predictions = search.predict_with(model, queries, extended_records, item_features)

    answers = [" ".join(item_ids[positions]) for positions in predictions]
    answer = pd.DataFrame({"query_id": queries["query_id"], "answer": answers})
    answer.to_csv("answer.csv", index=False, encoding="utf-8")
    validate("answer.csv")
    
    print("Ответ BM25 готов.", flush=True)


if __name__ == "__main__":
    main()
