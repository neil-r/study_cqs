import time
import typing

from . import data_model as event_extraction_task
from .. import prompt as prompt
from .. import db as db
from .. import discussion as discussion

def conduct_evaluations(
  tasks: typing.Iterator[event_extraction_task.EventExtractionTask],
  prompt_factory: prompt.PromptFactory,
  database: db.DatabaseSqlLite,
  discussion_model_factory: discussion.DiscussionStrategyFactory
):
  for task in tasks:
    if len(task.synset_options) <= 1:
       continue
    try:
        prompt_handler = prompt_factory.generate_prompt(task)
        prompt_strategy_name = prompt_handler.__class__.__name__

        # first check to ensure not already in database
        if database.has_task_result(
            task,
            prompt_strategy_name,
            model_id=discussion_model_factory.model_id
        ):
            print("Skipping task since it is already in database")
            continue
        

        start = time.time()
        
        d_model = discussion_model_factory.create()
        prompt_txt = prompt_handler.content
        response = d_model.speak(prompt_txt)
        follow_on_prompt = prompt_handler.handle_response(response[0].content)
        while follow_on_prompt is not None:
            follow_on_response = d_model.speak(follow_on_prompt.content)
            follow_on_prompt = follow_on_prompt.handle_response(follow_on_response[0].content)
        end = time.time()

        database.add_task_result(
            task,
            log=d_model.discussion.to_json(),
            prompt_strategy=prompt_strategy_name,
            model_id=discussion_model_factory.model_id,
            answer_value=prompt_handler.answer_value,
            discussion_duration=end-start,
            answer_response=prompt_handler.answer_response,
            correct=prompt_handler.answer_value==prompt_handler.answer_response,
        )
    except ValueError as e:
        print(f"Error handling {str(task)}, {str(e)}")
