"""Compare BM25 candidate retrieval with the submitted TF-IDF pipeline."""
import ast
import json
import os
import re
import time

import joblib
import nbformat
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits

OUTPUT_DIR = 'outputs/bm25'
os.makedirs(OUTPUT_DIR, exist_ok=True)
STARTED = time.monotonic()
TOP_K = 50
CHANNEL_SIZE = 100
BATCH_SIZE = 32
K1 = 1.2
B = 0.75


def load_notebook_functions(names):
    """Use the same cleaning, retrieval, features and Recall as the submitted code."""
    notebook = nbformat.read('solution.ipynb', 4)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        functions = [node for node in ast.parse(cell.source).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if functions:
            module = ast.Module(body=functions, type_ignores=[])
            exec(compile(module, '<solution-functions>', 'exec'), globals())


def build_bm25(texts, vocabulary):
    """Build a sparse BM25 matrix; rows are ads and columns are words."""
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


def retrieve_bm25(table, base_records):
    """Return BM25 lists and old candidates enlarged with BM25 hits."""
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


def predict_with(model, table, records):
    """Select 50 ads using exactly the submitted 18 model features."""
    predictions = []
    with threadpool_limits(limits=4):
        for number in range(len(table)):
            record = records[number]
            positions = record['positions']
            if not len(positions):
                predictions.append(positions)
                continue
            features = make_features(table.iloc[number], record)
            scores = model.predict_proba(features)[:, 1]
            order = np.argsort(-scores, kind='stable')[:TOP_K]
            predictions.append(positions[order])
    return predictions


def measure(table, predictions):
    return evaluate_lists(table, predictions)[0]


def run():
    global clean_text, top_positions, retrieve_candidates, make_features
    global evaluate_lists, recall_for_query, item_ids, item_locations, item_categories
    global word_vectorizer, char_vectorizer, word_vectors, char_vectors
    global location_positions, bm25_vectorizer, bm25_vectors, parameter_tokens
    global item_ratings, item_reviews, item_prices, phone_hidden, message_forbidden

    load_notebook_functions({'clean_text', 'top_positions', 'retrieve_candidates',
                             'make_features', 'recall_for_query', 'evaluate_lists'})
    with open('outputs/v3/run_status.json', encoding='utf-8') as file:
        v3_status = json.load(file)['status']
    prior = joblib.load('outputs/v2/candidates.joblib')
    if v3_status == 'complete':
        experiment = joblib.load('outputs/v3/candidates.joblib')
        dev_part = experiment['dev_part']
        test_part = experiment['test_part']
    else:
        print('V3 failed; using existing v2 dev and fresh test split.', flush=True)
        dev_part = prior['dev_part']
        test_part = prior['test_part']
    dev_records = prior['dev_records']
    assert list(dev_part['clean_query']) == list(prior['dev_part']['clean_query'])

    item_columns = ['item_id', 'item_title_raw', 'item_infm_params_text',
                    'item_description_raw', 'item_category_id', 'item_location_id',
                    'item_rating', 'item_rating_reviews_count', 'item_price',
                    'item_is_phone_hidden', 'item_is_message_forbidden']
    items = pd.read_parquet('data/benchmark_items.parquet', columns=item_columns)
    item_ids = items['item_id'].to_numpy()
    item_locations = items['item_location_id'].to_numpy()
    item_categories = items['item_category_id'].to_numpy()
    location_positions = items.groupby('item_location_id').indices
    items['clean_parameters'] = items['item_infm_params_text'].map(clean_text)
    parameter_tokens = [set(text.split()) for text in items['clean_parameters']]
    item_ratings = pd.to_numeric(items['item_rating'], errors='coerce').to_numpy(dtype=float)
    item_reviews = pd.to_numeric(items['item_rating_reviews_count'], errors='coerce').to_numpy(dtype=float)
    item_prices = pd.to_numeric(items['item_price'], errors='coerce').to_numpy(dtype=float)
    phone_hidden = items['item_is_phone_hidden'].fillna(False).to_numpy(dtype=float)
    message_forbidden = items['item_is_message_forbidden'].fillna(False).to_numpy(dtype=float)

    cached = joblib.load('outputs/v2/indexes.joblib')
    word_vectorizer = cached['word_vectorizer']
    char_vectorizer = cached['char_vectorizer']
    word_vectors = cached['word_vectors']
    char_vectors = cached['char_vectors']
    assert word_vectors.shape[0] == len(items) == char_vectors.shape[0]
    print('Loaded TF-IDF indexes.', flush=True)

    titles = items['item_title_raw'].map(clean_text)
    descriptions = items['item_description_raw'].str.slice(0, 1500).map(clean_text)
    corpus_text = titles + ' ' + titles + ' ' + items['clean_parameters'] + ' ' + descriptions
    bm25_vectorizer, bm25_vectors = build_bm25(corpus_text, word_vectorizer.vocabulary_)
    del titles, descriptions, corpus_text
    joblib.dump({'vectorizer': bm25_vectorizer, 'matrix': bm25_vectors,
                 'k1': K1, 'b': B}, OUTPUT_DIR + '/index.joblib')
    print('BM25 index saved.', flush=True)

    # Recompute the original candidates only for the fresh test split.
    test_records = retrieve_candidates(test_part)
    print('Fresh test original candidates computed.', flush=True)
    dev_global, dev_local, dev_extended = retrieve_bm25(dev_part, dev_records)
    test_global, test_local, test_extended = retrieve_bm25(test_part, test_records)

    previous_model = joblib.load('outputs/v1_backup/selector.joblib')['model']
    new_model = None
    if v3_status == 'complete':
        new_model = joblib.load('outputs/v3/model_10000.joblib')['model']

    rows = []
    for name, table, original, global_list, local_list, extended in [
        ('dev', dev_part, dev_records, dev_global, dev_local, dev_extended),
        ('fresh_test', test_part, test_records, test_global, test_local, test_extended),
    ]:
        row = {'split': name, 'queries': len(table),
               'bm25_global_recall50': measure(table, global_list),
               'bm25_local_recall50': measure(table, local_list),
               'original_candidate_recall': measure(table, [r['positions'] for r in original]),
               'with_bm25_candidate_recall': measure(table, [r['positions'] for r in extended]),
               'previous_model_original_recall50': measure(table,
                   predict_with(previous_model, table, original)),
               'previous_model_with_bm25_recall50': measure(table,
                   predict_with(previous_model, table, extended))}
        if new_model is not None:
            row['model_10000_original_recall50'] = measure(table,
                predict_with(new_model, table, original))
            row['model_10000_with_bm25_recall50'] = measure(table,
                predict_with(new_model, table, extended))
        rows.append(row)
        print('RESULT', json.dumps(row, ensure_ascii=False), flush=True)

    results = {'settings': {'k1': K1, 'b': B, 'channel_size': CHANNEL_SIZE,
                            'same_tfidf_vocab': True},
               'comparison': rows,
               'elapsed_seconds': round(time.monotonic() - STARTED, 1),
               'platform_recall_current_answer': 0.717977,
               'current_answer_untouched': True}
    with open(OUTPUT_DIR + '/metrics.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print('BM25_EXPERIMENT_OK', flush=True)


if __name__ == '__main__':
    run()
