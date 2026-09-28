"""Run every notebook cell and save actual outputs for review."""
import time

import nbformat
from nbclient import NotebookClient


def report_cell(cell, cell_index, **kwargs):
    """Показывает номер ячейки перед выполнением её кода."""
    if cell.cell_type == "code":
        print("Running cell", cell_index, flush=True)


def report_output(cell, cell_index, **kwargs):
    """Печатает текстовый вывод завершённой ячейки в терминал."""
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            print(output.text, end="", flush=True)


if __name__ == "__main__":
    # Читаем ноутбук и запускаем его с новым ядром Python.
    notebook = nbformat.read("solution.ipynb", as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        on_cell_start=report_cell,
        on_cell_executed=report_output,
    )
    started = time.time()
    try:
        client.execute()
    finally:
        # Сохраняем выполненные ячейки, даже если одна из них завершилась ошибкой.
        nbformat.write(notebook, "solution.ipynb")
    print("NOTEBOOK_OK", round(time.time() - started, 1), flush=True)
