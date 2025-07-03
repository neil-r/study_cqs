import typing

from driver_10_setup import datasets
from study_llm.event_extraction_task.dataset_rams import read_in_dataset, RamsPassage

from study_llm.full_text_search_db import FullTextSearch


search_db = FullTextSearch(db_name="fts_rams_search.db")

for dataset_file_path in datasets:
    extraction_tasks: typing.List[RamsPassage] = read_in_dataset(dataset_file_path)

    for et in extraction_tasks:
        search_db.insert_document(et.title(), et.passage_to_str())
