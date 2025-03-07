import time
import typing

from . import dataset_maven
from .. import prompt as prompt
from .. import db as db
from .. import discussion as discussion


def conduct_evaluations(
  tasks: typing.Iterator[dataset_maven.MavenParagraph],
  prompt_factory: prompt.PromptFactory,
  database: db.DatabaseSqlLite,
  d_model: discussion.DiscussionStrategy,
  determine_correct_f = lambda expected_value, actual_value : expected_value == actual_value
):
  for task in tasks:
    # if len(task.synset_options) <= 1:
    #    continue
    try:
        prompt_handler = prompt_factory.generate_prompt(task)
        prompt_strategy_name = prompt_handler.__class__.__name__

        # first check to ensure not already in database
        if database.has_task_result(
            task,
            prompt_strategy_name,
            model_id=d_model.model_id
        ):
            print("Skipping task since it is already in database")
            continue

        print("Starting discussion.")
        start = time.time()
        
        d = discussion.Discussion()
        
        prompt_txt = prompt_handler.content

        response = d_model.speak(d, prompt_txt)
        
        follow_on_prompt = prompt_handler.handle_response(response[0].content)
        while follow_on_prompt is not None:
            follow_on_response = d_model.speak(d, follow_on_prompt.content)
            follow_on_prompt = follow_on_prompt.handle_response(follow_on_response[0].content)
        end = time.time()
        print(f"Finished discussion ({end-start} seconds).")

        database.add_task_result(
            task,
            log=d.to_json(),
            prompt_strategy=prompt_strategy_name,
            model_id=d_model.model_id,
            answer_value=prompt_handler.answer_value,
            discussion_duration=end-start,
            answer_response=prompt_handler.answer_response,
            correct=determine_correct_f(prompt_handler.answer_value, prompt_handler.answer_response),
        )
    except ValueError as e:
        print(f"Error handling {str(task)}, {str(e)}")
