import typing

from driver_10_setup import datasets, keywords, determine_classification
from driver_11_prep_db import database
from study_llm.event_extraction_task.dataset_rams import read_in_dataset, RamsPassage

from study_llm.full_text_search_db import FullTextSearch


# perform keyword expansion using wordnet


search_db = FullTextSearch(db_name="fts_rams_search.db")

results = search_db.search(" OR ".join(keywords))

print(f"Found {len(results)} results")

result_map = {}
for r in results:
    title = r[0]
    if title in result_map:
        raise ValueError(
            "Didn't expect multiple pieces of content with the same title for the computations below"
        )

    result_map[title] = r

found_count = 0
unfound_map = {}

for dataset_file_path in datasets:
    extraction_tasks: typing.List[RamsPassage] = read_in_dataset(dataset_file_path)

    for et in extraction_tasks:

        if et.title() in result_map:
            found_count += 1
            result_map[et.title()] = et
        else:
            unfound_map[et.title()] = et

print(f"Re-mapped to extraction task structure count: {found_count}")

if __name__ == "__main__":
    model_text_search_id = "text_search"
    database.add_model_entry(
        model_id=model_text_search_id,
        model_name="Full Text Search",
        model_size=0,
        model_version=1,
    )

    for _, et in result_map.items():
        database.add_task_result(
            task=et,
            log="search",
            prompt_strategy="full_text_search",
            model_id=model_text_search_id,
            discussion_duration=0.0,
            answer_value="1" if determine_classification(et) else "0",
            answer_response="1",
            correct=determine_classification(et),
        )
    for _, et in unfound_map.items():
        database.add_task_result(
            task=et,
            log="search",
            prompt_strategy="full_text_search",
            model_id=model_text_search_id,
            discussion_duration=0.0,
            answer_value="1" if determine_classification(et) else "0",
            answer_response="0",
            correct=not determine_classification(et),
        )
