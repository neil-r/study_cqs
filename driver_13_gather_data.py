import typing


import study_llm.db as db
from study_llm.event_extraction_task.conduct_evaluations import conduct_evaluations
from study_llm.event_extraction_task.dataset_rams import (
    read_in_dataset,
    RamsPassage,
)
from study_llm import testing_model
from study_llm.hugging_face import HuggingFaceDiscussionStrategyFactory
from study_llm.open_ai import OpenAIStrategyFactory

from driver_10_setup import datasets, p_factories, huggingface_models
from driver_11_prep_db import database


# Prepare the model(s) that reads and responds to the prompt
dm_factories = [
    #  testing_model.SimpleDiscussionStrategyFactory(),
    #   openai_model.OpenAiDiscussionStrategyFactory(model="gpt-3.5-turbo-1106"),
    OpenAIStrategyFactory(model_name="gpt-4o-mini")
    # HuggingFaceDiscussionStrategyFactory(model_name=huggingface_models[0])
]


# read in datasets
extraction_tasks = []

for dataset_file_path in datasets:
    extraction_tasks: typing.List[RamsPassage] = read_in_dataset(dataset_file_path)

    print(
        f"The number of tasks is {len(list(x for x in extraction_tasks if len(x.events) > 0))}"
    )

    for discussion_model_factory in dm_factories:
        d_model = discussion_model_factory.create()
        for prompt_factory in p_factories:
            conduct_evaluations(
                extraction_tasks,
                prompt_factory,
                database,
                d_model,
                prompt_factory.create_assessor_func(),
            )
