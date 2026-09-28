"""Проверка CSV перед отправкой на платформу.

Пример: python validate_answer.py answer_v3.csv
"""

import re
import sys

import pandas as pd


def validate(filename):
    """Проверяет структуру файла и все идентификаторы по исходным Parquet."""
    queries = pd.read_parquet("data/benchmark_queries.parquet", columns=["query_id"])
    items = pd.read_parquet("data/benchmark_items.parquet", columns=["item_id"])
    answer = pd.read_csv(filename, dtype=str, keep_default_na=False)

    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("Нужны ровно две колонки: query_id, answer")

    if len(answer) != len(queries):
        raise ValueError("Число строк не совпадает с числом запросов")

    if answer["query_id"].duplicated().any():
        raise ValueError("Есть повторные query_id")

    if set(answer["query_id"]) != set(queries["query_id"]):
        raise ValueError("Набор query_id отличается от бенчмарка")

    if not answer["query_id"].map(lambda x: len(x) == 16).all():
        raise ValueError("Длина query_id должна быть 16 символов")

    valid_items = set(items["item_id"])
    lengths = []

    for query_id, value in answer[["query_id", "answer"]].itertuples(index=False):
        ids = value.split() if value else []

        if len(ids) > 50:
            raise ValueError(f"У {query_id} больше 50 объявлений")

        if len(ids) != len(set(ids)):
            raise ValueError(f"У {query_id} повторяются объявления")

        for item_id in ids:
            if not re.fullmatch(r"[0-9a-f]{16}", item_id):
                raise ValueError(f"Неверный формат item_id: {item_id}")

            if item_id not in valid_items:
                raise ValueError(f"Нет в корпусе: {item_id}")

        lengths.append(len(ids))

    print(f"OK: {filename}; запросов: {len(answer)}; "
          f"объявлений на запрос: от {min(lengths)} до {max(lengths)}")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "answer.csv")
