import typing

from driver_00_setup import event_types, datasets
from study_llm.event_extraction_task.dataset_maven import read_in_dataset, MavenParagraph

from study_llm.full_text_search_db import FullTextSearch


search_db = FullTextSearch()

for dataset_file_path in datasets:
    extraction_tasks:typing.List[MavenParagraph] = read_in_dataset(
        dataset_file_path,
        event_types=event_types
    )

    for et in extraction_tasks:
        search_db.insert_document(et.title(), " ".join(et.sentences))
