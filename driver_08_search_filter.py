import typing

from driver_00_setup import event_types, datasets, keywords
from study_llm.event_extraction_task.dataset_maven import read_in_dataset, MavenParagraph

from study_llm.full_text_search_db import FullTextSearch


# perform keyword expansion using wordnet


search_db = FullTextSearch()

results = search_db.search(" OR ".join(keywords))

print(results)
print(len(results))
