import typing
import random

from ..prompt import PromptFactory, Prompt
from .dataset_maven import MavenParagraph


def determine_true_false(response:str) -> typing.Optional[str]:
    response_l = response.lower()
    i_t = response_l.find("true")
    i_f = response_l.find("false")
    
    if i_t >= 0 and i_f >=0:
        return True if i_t < i_f else False
    elif i_t >= 0:
        return True
    elif i_f >= 0:
        return False
    return None



class DefaultEventExistsClassificationTaskPrompt(Prompt[MavenParagraph]):

    def __init__(self,
                 subject:MavenParagraph,
                 topic:str,
                 answer_value:bool,
                 response_interpretter_function=determine_true_false
        ):
        self.subject = subject
        self.topic = topic
        self.answer_value = answer_value
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:

        return f'''True or false, does the following paragraph describe one or more events about {self.topic}?

{self.subject.content}
'''
    
    def handle_response(self, response:str) -> typing.Optional["Prompt[MavenParagraph]"]:

        r = self.response_interpretter_function(response)
        if r is not None:
            self.answer_response = r

        return None


class DefaultEventExistsClassificationTaskPromptFactory(PromptFactory[MavenParagraph]):

    def __init__(self,
                 topic_msg:str,
        ):
            self.topic_msg = topic_msg

    def generate_prompt(self, topic:MavenParagraph) -> Prompt[MavenParagraph]:
        return DefaultEventExistsClassificationTaskPrompt(
            topic,
            self.topic_msg,
            answer_value=len(topic.events) > 0
        )
    


class DefaultExtractEventsPrompt(Prompt[MavenParagraph]):

    def __init__(self,
                 subject:MavenParagraph,
                 topic:str,
                 should_have_events:bool,
                 response_interpretter_function=determine_true_false
        ):
        self.subject = subject
        self.topic = topic
        self.should_have_events = should_have_events
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:

        return f'''Consider the following paragraph.

{self.subject.content}

Extract events about {self.topic}. If no releveant events exist, return an empty JSON list. Otherwise return a JSON list with objects for each event identifying the type of event and the partcipants. 
'''
    
    def handle_response(self, response:str) -> typing.Optional["Prompt[MavenParagraph]"]:

        r = self.response_interpretter_function(response)
        if r is not None:
            self.answer_response = r

        return None


class DefaultExtractEventsPromptFactory(PromptFactory[MavenParagraph]):

    def __init__(self,
                 topic_msg:str,
        ):
            self.topic_msg = topic_msg

    def generate_prompt(self, topic:MavenParagraph) -> Prompt[MavenParagraph]:
        return DefaultExtractEventsPrompt(
            topic,
            self.topic_msg,
            should_have_events=len(topic.events) > 0
        )
