import typing


import study_llm.db as db
from study_llm.event_extraction_task.conduct_evaluations import conduct_evaluations
from study_llm.event_extraction_task.dataset_maven import read_in_dataset, MavenParagraph
from study_llm import testing_model
from study_llm.hugging_face import HuggingFaceDiscussionStrategyFactory
from study_llm.open_ai import OpenAIStrategyFactory

from driver_00_setup import datasets, event_types, p_factories, huggingface_models
from driver_01_prep_db import database


# Prepare the model(s) that reads and responds to the prompt
dm_factories = [
#  testing_model.SimpleDiscussionStrategyFactory(),
#   openai_model.OpenAiDiscussionStrategyFactory(model="gpt-3.5-turbo-1106"),
   OpenAIStrategyFactory(model_name="gpt-4o-mini")
#   HuggingFaceDiscussionStrategyFactory(model_name=model_id)
]


# read in datasets
extraction_tasks = []

for dataset_file_path in datasets:
    extraction_tasks:typing.List[MavenParagraph] = read_in_dataset(
        dataset_file_path,
        event_types=event_types
    )

    print(f"The number of tasks is {len(list(x for x in extraction_tasks if len(x.events) > 0))}")

    for discussion_model_factory in dm_factories:
        d_model = discussion_model_factory.create()
        for prompt_factory in p_factories:
            conduct_evaluations(
                extraction_tasks[0:2],  # only conduct 3 tests per model and prompt
                prompt_factory,
                database,
                d_model,
                prompt_factory.create_assessor_func()
            )
