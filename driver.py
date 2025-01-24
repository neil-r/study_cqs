import typing

from study_llm.event_extraction_task.data_model import EventExtractionTask
import study_llm.event_extraction_task.prompts as prompts
import study_llm.db as db
from study_llm.event_extraction_task.conduct_evaluations import conduct_evaluations
from study_llm.event_extraction_task.dataset_maven import read_in_dataset

# configuration values
database = []


# Prepare the database that will store the discussion results
database = db.DatabaseSqlLite()

# Prepare the creator(s) of the prompts
p_factories = [
  #prompts.DefaultWsePromptFactory(),
  #prompts.DirectWsePromptFactory(),
  #prompts.RandomWsePromptFactory(),
  #prompts.OtherWsePromptFactory(),
]

# Prepare the model(s) that reads and responds to the prompt
dm_factories = [
  # testing_model.SimpleDiscussionStrategyFactory(),
  # t5_model.T5DiscussionStrategyFactory(),
  # guanaco_7B_model.GuanacoDiscussionStrategyFactory(),
  # vicuna_7B_model.VicunaDiscussionStrategyFactory(),
  # palm_model.PalmDiscussionStrategyFactory(),
  # llama2_13B_model.Llama2_13BDiscussionStrategyFactory(),
  # llama2_7B_model.Llama2_7BDiscussionStrategyFactory(),
  # openai_model.OpenAiDiscussionStrategyFactory(model="gpt-3.5-turbo-1106"),
  #openai_model.OpenAiDiscussionStrategyFactory(model="gpt-4-0613")
]

# setup models
models = []

# setup 
datasets = [
    "data/maven/train.jsonl"
]

types = [
    "Violence", "Surrounding", "Besieging", "Attack", "Military_operation", "Hostile_encounter", "Terrorism", "Bearing_arms", "Defending", "Killing"
]

# read in datasets
extraction_tasks = []

for dataset_file_path in datasets:
    extraction_tasks:typing.List[EventExtractionTask] = read_in_dataset(
        dataset_file_path,
        types=types
    )

    print(f"The number of tasks is {len(list(x for x in extraction_tasks if len(x.events) > 0))}")

    for prompt_factory in p_factories:
        for discussion_model_factory in dm_factories:
            conduct_evaluations(extraction_tasks, prompt_factory, database, discussion_model_factory)

