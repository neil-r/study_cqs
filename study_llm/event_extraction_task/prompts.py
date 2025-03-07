import typing
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


def extract_json_list(text:str) -> str | None:
    """
    Extracts the first JSON list found in the given text.
    
    Args:
        text (str): The input string containing a JSON list.
    
    Returns:
        list: The extracted JSON list or None if no valid list is found.
    """
    start = text.find('[')
    if start == -1:
        return None  # No opening bracket found

    end = text.rfind(']')
    if end == -1:
        return None  # No closing bracket found

    json_str = text[start:end+1]
    try:
        return json.dumps(json.loads(json_str))
    except json.JSONDecodeError:
        return None  # Invalid JSON list

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
            self.answer_response = self.response_interpretter_function(response)
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
    
    def create_assessor_func(self):
        return lambda expected_value, actual_value : expected_value == actual_value

    

class DefaultExtractEventsPrompt(Prompt[MavenParagraph]):

    def __init__(self,
                 subject: MavenParagraph,
                 topic: str,
                 answer_value: bool,
                 response_interpretter_function=extract_json_list
        ):
        self.subject = subject
        self.topic = topic
        self.answer_value = answer_value
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:
        return f'''Consider the following paragraph.\n\n{self.subject.content}\n\nIdentify events about {self.topic}. For each event, return a JSON object with an attribute specifying the event type and, if stated, the following event meta-data as additional attributes: the cause of the event with a "cause" attribute and the datetime of the event with a "date" attribute. If no relevant events exist, return an empty JSON list. Otherwise return a JSON list with the JSON objects for each event.\n'''
    
    def handle_response(self, response: str) -> typing.Optional["Prompt[MavenParagraph]"]:
        try:
            self.answer_response = self.response_interpretter_function(response)
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
    
    def create_assessor_func(self):
        return lambda expected_value, actual_value : expected_value == non_empty_json_list(actual_value)


def non_empty_json_list(json_str) -> bool:
    if json_str is None:
        return False
    jv = json.loads(json_str)
    if isinstance(jv, list):
        return len(jv) > 0
    return False

