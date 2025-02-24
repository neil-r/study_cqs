
import pytest
from unittest.mock import Mock
from study_llm.event_extraction_task.prompts import DefaultEventExistsClassificationTaskPrompt, DefaultExtractEventsPrompt

@pytest.fixture
def sample_paragraph():
    class MavenParagraph:
        def __init__(self, content):
            self.content = content
    return MavenParagraph("This is a sample paragraph describing an event about fighting wars.")

@pytest.fixture
def mock_response_interpretter():
    return Mock(return_value=True)

@pytest.fixture
def mock_json_response_interpretter():
    return Mock(return_value=[{"event": "battle", "participants": ["ally", "enemy"]}])


def test_event_exists_classification_prompt_content(sample_paragraph):
    prompt = DefaultEventExistsClassificationTaskPrompt(
        subject=sample_paragraph,
        topic="fighting wars",
        answer_value=True
    )
    expected_content = (
        "Only respond with either True or false, does the following paragraph describe one or more events about fighting wars?\n\nThis is a sample paragraph describing an event about fighting wars.\n"
    )
    # expected_content = "Junk"

    assert prompt.content == expected_content



def test_event_exists_classification_handle_response(sample_paragraph, mock_response_interpretter):
    prompt = DefaultEventExistsClassificationTaskPrompt(
        subject=sample_paragraph,
        topic="fighting wars",
        answer_value=True,
        response_interpretter_function=mock_response_interpretter
    )
    prompt.handle_response("True")
    assert prompt.answer_response is True
    mock_response_interpretter.assert_called_once_with("True")


def test_extract_events_prompt_content(sample_paragraph):
    prompt = DefaultExtractEventsPrompt(
        subject=sample_paragraph,
        topic="fighting wars",
        answer_value=True
    )
    expected_content = (
        '''Consider the following paragraph.\n\nThis is a sample paragraph describing an event about fighting wars.\n\nExtract events about fighting wars. If no relevant events exist, return an empty JSON list. Otherwise return a JSON list with objects for each event identifying the type of event and the participants.\n'''
    )
    # expected_content = "Junk"
    print(prompt.content)
    print(expected_content)
    assert prompt.content == expected_content


def test_extract_events_prompt_handle_response(sample_paragraph, mock_json_response_interpretter):
    prompt = DefaultExtractEventsPrompt(
        subject=sample_paragraph,
        topic="fighting wars",
        answer_value=True,
        response_interpretter_function=mock_json_response_interpretter
    )
    prompt.handle_response("[{\"event\": \"battle\", \"participants\": [\"ally\", \"enemy\"]}]")
    # prompt.handle_response("[]")
    assert prompt.answer_response == [{"event": "battle", "participants": ["ally", "enemy"]}]
    mock_json_response_interpretter.assert_called_once_with("[{\"event\": \"battle\", \"participants\": [\"ally\", \"enemy\"]}]")
