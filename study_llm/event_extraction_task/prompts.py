import typing
import random
import json

from study_llm.prompt import PromptFactory, Prompt
from .dataset_maven import MavenParagraph

def determine_true_false(response: str) -> typing.Optional[str]:
    try:
        response_l = response.lower()
        i_t = response_l.find("true")
        i_f = response_l.find("false")
        
        if i_t >= 0 and i_f >= 0:
            return True if i_t < i_f else False
        elif i_t >= 0:
            return True
        elif i_f >= 0:
            return False
        return None
    except Exception as e:
        print(f"Error in determine_true_false: {e}")
        return None

def determine_json_response(response: str) -> typing.Optional[str]:
    try:
        if response == '[]':
            return False
        parsed_response = json.loads(response)  # Try parsing as JSON
        return isinstance(parsed_response, (dict, list))  # Check if it's a valid JSON type
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None  # Not a valid JSON response
    except Exception as e:
        print(f"Error in determine_json_response: {e}")
        return None

class DefaultEventExistsClassificationTaskPrompt(Prompt[MavenParagraph]):

    def __init__(self,
                 subject: MavenParagraph,
                 topic: str,
                 answer_value: bool,
                 response_interpretter_function=determine_true_false
        ):
        self.subject = subject
        self.topic = topic
        self.answer_value = answer_value
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:
        return f'''Only respond with either True or false, does the following paragraph describe one or more events about {self.topic}?\n\n{self.subject.content}\n'''
    
    def handle_response(self, response: str) -> typing.Optional["Prompt[MavenParagraph]"]:
        # # Test Failure Case:
        # r = False
        # self.answer_response = r
        # return None
    
        # Actual Case        
        try:
            r = self.response_interpretter_function(response)
            if r is not None:
                self.answer_response = r
        except Exception as e:
            print(f"Error handling response: {e}")
        return None


class DefaultEventExistsClassificationTaskPromptFactory(PromptFactory[MavenParagraph]):

    def __init__(self,
                 topic_msg: str,
        ):
        self.topic_msg = topic_msg

    def generate_prompt(self, topic: MavenParagraph) -> Prompt[MavenParagraph]:
        return DefaultEventExistsClassificationTaskPrompt(
            topic,
            self.topic_msg,
            answer_value=len(topic.events) > 0
        )
    

class DefaultExtractEventsPrompt(Prompt[MavenParagraph]):

    def __init__(self,
                 subject: MavenParagraph,
                 topic: str,
                 answer_value: bool,
                 response_interpretter_function=determine_json_response
        ):
        self.subject = subject
        self.topic = topic
        self.answer_value = answer_value
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:
        return f'''Consider the following paragraph.\n\n{self.subject.content}\n\nExtract events about {self.topic}. If no relevant events exist, return an empty JSON list. Otherwise return a JSON list with objects for each event identifying the type of event and the participants.\n'''
    
    def handle_response(self, response: str) -> typing.Optional["Prompt[MavenParagraph]"]:
        try:
            r = self.response_interpretter_function(response)
            if r is not None:
                self.answer_response = r
        except Exception as e:
            print(f"Error handling response in DefaultExtractEventsPrompt: {e}")
        return None


class DefaultExtractEventsPromptFactory(PromptFactory[MavenParagraph]):

    def __init__(self,
                 topic_msg: str,
        ):
        self.topic_msg = topic_msg

    def generate_prompt(self, topic: MavenParagraph) -> Prompt[MavenParagraph]:
        return DefaultExtractEventsPrompt(
            topic,
            self.topic_msg,
            answer_value=len(topic.events) > 0
        )
