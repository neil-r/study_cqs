import typing


import study_llm.db as db
from study_llm.event_extraction_task.conduct_evaluations import conduct_evaluations
from study_llm.event_extraction_task.dataset_maven import read_in_dataset, MavenParagraph
from study_llm import testing_model
from study_llm.hugging_face import HuggingFaceDiscussionStrategyFactory

from driver_00_setup import datasets, event_types, p_factories, huggingface_models
from driver_01_prep_db import database


# Prepare the model(s) that reads and responds to the prompt
dm_factories = [
#  testing_model.SimpleDiscussionStrategyFactory(),
#   t5_model.T5DiscussionStrategyFactory(),
  # guanaco_7B_model.GuanacoDiscussionStrategyFactory(),
  # vicuna_7B_model.VicunaDiscussionStrategyFactory(),
#   palm_model.PalmDiscussionStrategyFactory(),
#   llama2_13B_model.Llama2_13BDiscussionStrategyFactory(),
  # llama2_7B_model.Llama2_7BDiscussionStrategyFactory(),
  # openai_model.OpenAiDiscussionStrategyFactory(model="gpt-3.5-turbo-1106"),
#   openai_model.OpenAiDiscussionStrategyFactory(model="gpt-4-0613")
]

for model_id in huggingface_models:
    dm_factories.append(HuggingFaceDiscussionStrategyFactory(model_name=model_id))


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
                extraction_tasks,  # only conduct 3 tests per model and prompt
                prompt_factory,
                database,
                d_model,
                prompt_factory.create_assessor_func()
            )
