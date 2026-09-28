"""BM25: индекс, поиск кандидатов и отбор моделью."""

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits

TOP_K = 50
CHANNEL_SIZE = 100
BATCH_SIZE = 32
K1 = 1.2
B = 0.75

def clean_text(text):
    """Приводит текст к нижнему регистру и убирает лишние символы."""

    if pd.isna(text):
        return ""

    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def top_positions(scores, limit):
    """Возвращает позиции до limit результатов с положительными оценками."""
    positions = np.flatnonzero(scores > 0)
    order = np.argsort(-scores[positions], kind="stable")[:limit]
    return positions[order]


def retrieve_candidates(table, tfidf, location_positions):
    """Собирает четыре списка TF-IDF, как в основном ноутбуке."""
    word_vectorizer = tfidf["word_vectorizer"]
    char_vectorizer = tfidf["char_vectorizer"]
    word_vectors = tfidf["word_vectors"]
    char_vectors = tfidf["char_vectors"]
    records = []

    for start in range(0, len(table), BATCH_SIZE):
        batch = table.iloc[start:start + BATCH_SIZE]
        texts = []

        for _, row in batch.iterrows():
            text = clean_text(row["search_query"])

            if not text:
                text = clean_text(row["search_infm_params_text"])

            texts.append(text)

        word_scores = (word_vectorizer.transform(texts) @ word_vectors.T).toarray()
        char_scores = (char_vectorizer.transform(texts) @ char_vectors.T).toarray()

        for number in range(len(batch)):
            row = batch.iloc[number]
            words = word_scores[number]
            chars = char_scores[number]
            word_best = top_positions(words, CHANNEL_SIZE)
            char_best = top_positions(chars, CHANNEL_SIZE)
            local = location_positions.get(row["search_location_id"], np.array([], dtype=int))
            local_word_best = local[top_positions(words[local], CHANNEL_SIZE)]
            local_char_best = local[top_positions(chars[local], CHANNEL_SIZE)]
            positions = np.unique(np.concatenate([
                word_best, char_best, local_word_best, local_char_best,
            ]))
            records.append({
                "positions": positions,
                "word_score": words[positions],
                "char_score": chars[positions],
                "baseline": word_best[:TOP_K],
            })

        if (start // BATCH_SIZE + 1) % 20 == 0:
            print("TF-IDF: обработано запросов:", min(start + BATCH_SIZE, len(table)), flush=True)

    return records


def make_features(row, record, item_features):
    """Рассчитывает те же 18 признаков и в том же порядке, что в ноутбуке."""
    positions = record["positions"]
    count = len(positions)
    words = record["word_score"]
    chars = record["char_score"]
    same_location = item_features["locations"][positions] == row["search_location_id"]
    category_specified = pd.notna(row["search_category"]) and row["search_category"] != 0
    same_category = (item_features["categories"][positions] == row["search_category"]) & category_specified
    filter_words = set(clean_text(row["search_infm_params_text"]).split())
    overlap = []

    for position in positions:
        common = filter_words.intersection(item_features["parameter_tokens"][position])
        overlap.append(len(common) / max(1, len(filter_words)))

    ratings = item_features["ratings"][positions]
    reviews = item_features["reviews"][positions]
    prices = item_features["prices"][positions]
    delivery = row["search_is_delivery_search"]

    if pd.isna(delivery):
        delivery = 0

    features = np.column_stack([
        words, chars, words + chars, words * chars,
        same_location, same_category, np.full(count, category_specified),
        np.full(count, len(clean_text(row["search_query"]).split())),
        np.full(count, delivery), overlap,
        np.nan_to_num(ratings), np.isnan(ratings),
        np.log1p(np.maximum(np.nan_to_num(reviews), 0)), np.isnan(reviews),
        np.log1p(np.maximum(np.nan_to_num(prices), 0)), np.isnan(prices),
        item_features["phone_hidden"][positions], item_features["message_forbidden"][positions],
    ])
    return features.astype(np.float32)


def build_bm25(texts, vocabulary):
    """Строит индекс BM25: строки — объявления, столбцы — слова."""
    vectorizer = CountVectorizer(vocabulary=vocabulary, dtype=np.float32, binary=False)
    counts = vectorizer.transform(texts)

    document_count, term_count = counts.shape
    document_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = document_lengths.mean()
    document_frequency = counts.getnnz(axis=0)

    idf = np.log1p((document_count - document_frequency + 0.5)
                   / (document_frequency + 0.5)).astype(np.float32)

    row_ids = np.repeat(np.arange(document_count, dtype=np.int32),
                        np.diff(counts.indptr))
    
    term_frequency = counts.data
    length_factor = K1 * (1 - B + B * document_lengths[row_ids] / average_length)

    weighted = idf[counts.indices] * (K1 + 1) * term_frequency
    weighted = weighted / (term_frequency + length_factor)

    bm25 = csr_matrix((weighted.astype(np.float32), counts.indices, counts.indptr),
                      shape=(document_count, term_count))
    
    print('BM25 matrix:', bm25.shape, 'nonzero:', bm25.nnz, flush=True)

    del counts, row_ids, document_lengths, weighted
    return vectorizer, bm25.tocsc()


def retrieve_bm25(table, base_records, tfidf, bm25_index, location_positions, item_locations):
    """Добавляет глобальные и локальные результаты BM25 к TF-IDF-кандидатам."""
    
    word_vectorizer = tfidf["word_vectorizer"]
    char_vectorizer = tfidf["char_vectorizer"]
    word_vectors = tfidf["word_vectors"]
    char_vectors = tfidf["char_vectors"]
    bm25_vectorizer = bm25_index["vectorizer"]
    bm25_vectors = bm25_index["matrix"]

    assert len(table) == len(base_records)
    bm25_global = []
    bm25_local = []
    extended_records = []

    for start in range(0, len(table), BATCH_SIZE):
        batch = table.iloc[start:start + BATCH_SIZE]
        texts = []

        for _, row in batch.iterrows():
            text = clean_text(row['search_query'])

            if not text:
                text = clean_text(row['search_infm_params_text'])

            texts.append(text)

        bm_scores = (bm25_vectorizer.transform(texts) @ bm25_vectors.T).toarray()
        words = (word_vectorizer.transform(texts) @ word_vectors.T).toarray()
        chars = (char_vectorizer.transform(texts) @ char_vectors.T).toarray()

        for offset in range(len(batch)):
            number = start + offset
            row = batch.iloc[offset]
            scores = bm_scores[offset]
            location = location_positions.get(row['search_location_id'],
                                              np.array([], dtype=int))
            global_best = top_positions(scores, CHANNEL_SIZE)
            local_order = top_positions(scores[location], CHANNEL_SIZE)
            local_best = location[local_order]

            bm25_positions = np.unique(np.concatenate([global_best, local_best]))
            
            extended = np.union1d(base_records[number]['positions'], bm25_positions)
            extended_records.append({
                'positions': extended,
                'word_score': words[offset][extended],
                'char_score': chars[offset][extended],
                'baseline': base_records[number]['baseline'],
            })
            
            bm25_global.append(global_best[:TOP_K])
            boosted = scores[bm25_positions] * (
                1 + 2 * (item_locations[bm25_positions] == row['search_location_id']))
            order = np.argsort(-boosted, kind='stable')[:TOP_K]
            bm25_local.append(bm25_positions[order])

        if (start // BATCH_SIZE + 1) % 20 == 0:
            print('BM25 processed:', min(start + BATCH_SIZE, len(table)), flush=True)

    return bm25_global, bm25_local, extended_records


def predict_with(model, table, records, item_features):
    """Выбирает до 50 объявлений по оценкам обученной модели."""
    predictions = []

    with threadpool_limits(limits=4):
        for number in range(len(table)):
            record = records[number]
            positions = record['positions']

            if not len(positions):
                predictions.append(positions)
                continue

            features = make_features(table.iloc[number], record, item_features)
            scores = model.predict_proba(features)[:, 1]
            order = np.argsort(-scores, kind='stable')[:TOP_K]
            predictions.append(positions[order])

    return predictions

