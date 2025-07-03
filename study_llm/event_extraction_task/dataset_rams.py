import json
from dataclasses import dataclass
from typing import List, Tuple
import typing
import time
import typing
from .. import db as db
from .. import discussion as discussion

from study_llm.prompt import PromptFactory, Prompt


@dataclass
class RamsEvent:
    type_indicator: Tuple[str, float]
    trigger_offset: Tuple[int, int]
    components: List

    def __str__(self) -> str:
        return f"Type={self.type_indicator}-Offset={self.trigger_offset}-Components={self.components}"

    def to_json(self):
        return {
            "type_indicator": self.type_indicator,
            "trigger_offset": self.trigger_offset,
            "components": self.components,
        }


def get_text(sentences, start_offset, end_offset) -> str:
    parts = []

    running_offset = 0
    for s in sentences:
        l = len(s)
        end_sentence_offset = running_offset + l

        if start_offset >= running_offset and start_offset < end_sentence_offset:
            offset_offset = start_offset - running_offset
            for i in range(end_offset - start_offset + 1):
                parts.append(s[i + offset_offset])

        running_offset = end_sentence_offset

    return " ".join(parts)


def _replace_first_quote(s):
    if s[0:2] == "“ ":
        return "“" + s[2:]
    elif s[0:2] == '" ':
        return '"' + s[2:]
    else:
        return s


@dataclass
class RamsPassage:
    raw_json: dict
    events: List[RamsEvent]
    sentences: List[List[str]]

    def to_json(self):
        return {
            "raw_json": self.raw_json,
            "events": list(e.to_json() for e in self.events),
            "sentences": self.sentences,
        }

    def passage_to_str(self) -> str:
        return _replace_first_quote(
            (" ".join(" ".join(s) for s in self.sentences))
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ;", ";")
            .replace(" '", "'")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(' "', '"')
            .replace(" “ ", " “")
            .replace(" ” ", "” ")
            .replace(" - ", "-")
            .replace("( ", "(")
            .replace(" )", ")")
            .replace(" n’t", "n’t")
            .replace(" ’", "’")
            .replace("$ ", "$")
            .replace(" ”.", "”.")
            .replace("[ ", "[")
            .replace(" ]", "]")
            .replace(" ”,", "”,")
            .replace("\xad", "")
            .replace(" : ", ": ")
            .replace(" ; ", "; ")
        )

    @property
    def content(self):
        return self.passage_to_str()

    def title(self) -> str:
        return self.raw_json["doc_key"]

    def has_event_type(self, event_types):
        return (
            len(list(e for e in self.events if e.type_indicator[0] in event_types)) > 0
        )

    def __str__(self) -> str:
        parts = [
            "Events:" + ", ".join(str(e) for e in self.events),
            "Passage: " + self.passage_to_str(),
        ]

        return "\n\t".join(parts)


def read_in_dataset(file_path, event_types=None) -> List[RamsPassage]:

    annotated_paragraphs = []

    # Open the JSONL file and process each line
    with open(file_path, "r", errors="ignore", encoding="utf-8") as file:
        for line in file:
            # Load the JSON data from the line
            data = json.loads(line)

            # Print the data
            # print(data)
            ent_span_map = {}

            for ent_span in data["ent_spans"]:
                offset_1 = ent_span[0]
                offset_2 = ent_span[1]
                for type_indicator in ent_span[2]:
                    ent_span_map[f"{offset_1}-{offset_2}-{type_indicator[0]}"] = (
                        type_indicator[0],
                        type_indicator[1],
                    ) + (get_text(data["sentences"], offset_1, offset_2),)

            evt_link_map = {}

            for gold_evt_link in data["gold_evt_links"]:
                evt_trigger_offset_1, evt_trigger_offset_2 = gold_evt_link[0]
                ent_offset_1, ent_offset_2 = gold_evt_link[1]
                ent_type = gold_evt_link[2]
                ent_id = f"{ent_offset_1}-{ent_offset_2}-{ent_type}"

                event_id = f"{evt_trigger_offset_1}-{evt_trigger_offset_2}"

                if event_id not in evt_link_map:
                    evt_link_map[event_id] = set()

                evt_link_map[event_id].add(ent_id)

            annotated_paragraphs.append(
                RamsPassage(
                    raw_json=data,
                    sentences=data["sentences"],
                    events=list(
                        RamsEvent(
                            type_indicator=type_indicator,
                            trigger_offset=(base_evt_trigger[0], base_evt_trigger[1]),
                            components=(
                                list(
                                    ent_span_map[ent_link]
                                    for ent_link in evt_link_map[
                                        f"{base_evt_trigger[0]}-{base_evt_trigger[1]}"
                                    ]
                                )
                                if f"{base_evt_trigger[0]}-{base_evt_trigger[1]}"
                                in evt_link_map
                                else []
                            ),
                        )
                        for base_evt_trigger in data["evt_triggers"]
                        if event_types is None
                        or base_evt_trigger["type"] in event_types
                        for type_indicator in base_evt_trigger[2]
                    ),
                )
            )
            """
            {
                "rel_triggers": [],
                "gold_rel_links": [],
                "doc_key": "nw_RC02da7e59505d54b2e179a955ec897763412bbd89dba3a4c513b25e62",
                "ent_spans": [
                    [66, 68, [["evt034arg01participant", 1.0]]]
                ],
                "language_id": "eng",
                "source_url": "http://www.reuters.com/article/us-usa-election-trump-idUSKCN10M146",
                "evt_triggers": [[75, 75, [["contact.discussion.meet", 1.0]]]],
                "split": "test",
                "sentences":
                    [
                        ["Republicans", "frequently", "trace", "the", "birth", "of", "Islamic", "State", "to", "the", "Obama", "administration", "\u2019s", "decision", "to", "withdraw", "the", "last", "U.S.", "forces", "from", "Iraq", "by", "the", "end", "of", "2011", "."],
                        ["But", "many", "analysts", "argue", "its", "roots", "lie", "in", "the", "decision", "of", "George", "W.", "Bush", "\u2019s", "Republican", "administration", "to", "invade", "Iraq", "in", "2003", "without", "a", "plan", "to", "fill", "the", "vacuum", "created", "by", "Saddam", "Hussein", "\u2019s", "ouster", "."],
                        ["It", "was", "Bush", "\u2019s", "administration", ",", "not", "Obama", "\u2019s", ",", "that", "negotiated", "the", "2009", "agreement", "that", "called", "for", "the", "withdrawal", "of", "all", "U.S.", "forces", "from", "Iraq", "by", "Dec.", "31", ",", "2011", "."],
                        ["Clinton", "posted", "on", "Twitter", "that", "Trump", "'s", "comments", "are", "disqualifying", "."],
                        ["\"", "Anyone", "willing", "to", "sink", "so", "low", ",", "so", "often", "should", "never", "be", "allowed", "to", "serve", "as", "our", "commander", "-", "in", "-", "chief", ",", "\"", "she", "wrote", "."]
                    ],
                "gold_evt_links": [
                    [[75, 75], [66, 68], "evt034arg01participant"]
                ]
            }


            {
                "rel_triggers": [],
                "gold_rel_links": [],
                "doc_key": "nw_RC06f93fc9b12b98ac26b24fc14a957aaec925686625df43896942248e",
                "ent_spans": [],
                "language_id": "eng",
                "source_url": "http://www.theguardian.com/law/2016/mar/16/obama-nominates-merrick-garland-supreme-court-dc-appeals-court-judge",
                "evt_triggers": [
                    [105, 106, [["justice.judicialconsequences.execute", 1.0]]]
                ],
                "split": "test",
                "sentences": [
                    ["In", "the", "immediate", "aftermath", "of", "the", "Oklahoma", "City", "bombing", "that", "ripped", "apart", "a", "federal", "building", "and", "killed", "168", "people", ",", "he", "was", "dispatched", "to", "the", "city", "to", "set", "up", "the", "early", "stages", "of", "the", "prosecution", "case", ",", "winning", "plaudits", "for", "gathering", "large", "amounts", "of", "evidence", "that", "led", "to", "the", "convictions", "of", "both", "McVeigh", "and", "Terry", "Nichols", "."],
                    ["He", "was", "also", "central", "to", "the", "prosecution", "of", "the", "Unabomber", ",", "Ted", "Kaczynski", "."],
                    ["He", "acquired", "his", "passion", "for", "being", "a", "judge", "by", "sitting", "at", "the", "feet", "of", "the", "liberal", "supreme", "court", "justice", "William", "Brennan", ",", "who", "was", "a", "champion", "of", "progressive", "policies", "such", "as", "opposition", "to", "the", "death", "penalty", "and", "support", "of", "abortion", "rights", "."],
                    ["Garland", "has", "had", "plenty", "of", "opportunity", "to", "wield", "similar", "progressive", "influence", "as", "chief", "judge", "of", "the", "DC", "appeals", "circuit", "which", ",", "given", "its", "location", ",", "frequently", "acts", "as", "arbiter", "in", "major", "cases", "concerning", "the", "federal", "government", "."],
                    ["Among", "those", "were", "the", "2008", "judgment", "from", "the", "appeals", "court", ",", "led", "by", "Garland", ",", "that", "punched", "a", "hole", "in", "the", "Bush", "administration", "\u2019s", "detention", "of", "so", "-", "called", "\u201c", "enemy", "combatants", "\u201d", "in", "Guant\u00e1namo", "Bay", "without", "any", "oversight", "from", "the", "civilian", "courts", "."]
                ],
                "gold_evt_links": []
            }

            {
                "rel_triggers": [],
                "gold_rel_links": [],
                "doc_key": "nw_RC1f4c5e3ac366f6fc671d750bedba3909098c8797c3f5184799adda2c",
                "ent_spans": [[12, 12, [["evt087arg04place", 1.0]]]],
                "language_id": "eng",
                "source_url": "http://aranews.net/2016/07/iranian-regime-executes-dozens-opposition-activists/",
                "evt_triggers": [
                    [10, 10, [["justice.judicialconsequences.execute", 1.0]]]
                ],
                "split": "test",
                "sentences": [
                    ["A", "file", "photo", "showing", "Iranians", "while", "reacting", "to", "a", "public", "execution", "in", "Tehran", "."],
                    ["(", "AP", ")"],
                    ["Click", "to", "email", "this", "to", "a", "friend", "(", "Opens", "in", "new", "window", ")"]
                ],
                "gold_evt_links": [
                    [[10, 10], [12, 12], "evt087arg04place"]
                ]ent_link
            }
            {
                "rel_triggers": [],
                "gold_rel_links": [],
                "doc_key": "nw_RC2bbd630fe275996128ef4e99d76f9f297fdf01da65d71b52c7598822",
                "ent_spans": [
                    [71, 77, [["evt081arg04crime", 1.0]]],
                    [95, 95, [["evt081arg02defendant", 1.0]]]
                ],
                "language_id": "eng",
                "source_url": "http://www.theguardian.com/society/2016/apr/07/the-sugar-conspiracy-robert-lustig-john-yudkin",
                "evt_triggers": [
                    [69, 69, [["justice.initiatejudicialprocess.chargeindict", 1.0]]]
                ],
                "split": "test",
                "sentences": [
                    ["If", "ever", "there", "was", "a", "case", "that", "an", "information", "democracy", "is", "preferable", "to", "an", "information", "oligarchy", ",", "then", "this", "is", "it"],
                    ["The", "nutritional", "establishment", "has", "proved", "itself", ",", "over", "the", "years", ",", "skilled", "at", "ad", "hominem", "takedowns", ",", "but", "it", "is", "harder", "for", "them", "to", "do", "to", "Robert", "Lustig", "or", "Nina", "Teicholz", "what", "they", "once", "did", "to", "John", "Yudkin", "."], ["Harder", ",", "too", ",", "to", "deflect", "or", "smother", "the", "charge", "that", "the", "promotion", "of", "low", "-", "fat", "diets", "was", "a", "40-year", "fad", ",", "with", "disastrous", "outcomes", ",", "conceived", "of", ",", "authorised", ",", "and", "policed", "by", "nutritionists", "."], ["Professor", "John", "Yudkin", "retired", "from", "his", "post", "at", "Queen", "Elizabeth", "College", "in", "1971", ",", "to", "write", "Pure", ",", "White", "and", "Deadly", "."], ["The", "college", "reneged", "on", "a", "promise", "to", "allow", "him", "to", "continue", "to", "use", "its", "research", "facilities", "."]
                ],
                "gold_evt_links": [
                    [[69, 69], [95, 95], "evt081arg02defendant"],
                    [[69, 69], [71, 77], "evt081arg04crime"]
                ]
            }
            {"rel_triggers": [], "gold_rel_links": [], "doc_key": "nw_RC03a8d123c6167de26fb5bd617942f690d413dd14ec651b427f23be14", "ent_spans": [[40, 41, [["evt090arg02victim", 1.0]]], [76, 76, [["evt090arg04place", 1.0]]], [20, 21, [["evt090arg01killer", 1.0]]]], "language_id": "eng", "source_url": "http://www.theguardian.com/world/2016/feb/29/woman-detained-in-moscow-carrying-severed-head-of-toddler", "evt_triggers": [[25, 25, [["life.die.deathcausedbyviolentevents", 1.0]]]], "split": "test", "sentences": [["Investigators", "say", "nanny", "killed", "young", "girl", "and", "set", "flat", "on", "fire", "before", "being", "detained", "at", "metro", "station"], ["Police", "have", "detained", "a", "nanny", "on", "suspicion", "of", "murder", "after", "she", "was", "found", "at", "a", "Moscow", "metro", "station", "holding", "the", "severed", "head", "of", "a", "child", "."], ["The", "woman", ",", "believed", "to", "be", "from", "central", "Asia", ",", "can", "be", "seen", "in", "video", "footage", "holding", "up", "what", "appears", "to", "be", "a", "severed", "head", "near", "Oktyabrskoye", "Pole", "station", "in", "north", "-", "western", "Moscow", "."], ["In", "the", "video", "she", "is", "covered", "in", "black", "except", "for", "her", "face", "and", "she", "can", "be", "heard", "shouting", "\u201c", "for", "a", "terrorist", ",", "for", "your", "death", "\u201d", "."]], "gold_evt_links": [[[25, 25], [20, 21], "evt090arg01killer"], [[25, 25], [40, 41], "evt090arg02victim"], [[25, 25], [76, 76], "evt090arg04place"]]}
            """

    return annotated_paragraphs


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
        if response == "[]":
            return False
        parsed_response = json.loads(response)  # Try parsing as JSON
        return isinstance(
            parsed_response, (dict, list)
        )  # Check if it's a valid JSON type
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None  # Not a valid JSON response
    except Exception as e:
        print(f"Error in determine_json_response: {e}")
        return None


def extract_json_list(text: str) -> str | None:
    """
    Extracts the first JSON list found in the given text.

    Args:
        text (str): The input string containing a JSON list.

    Returns:
        list: The extracted JSON list or None if no valid list is found.
    """
    start = text.find("[")
    if start == -1:
        return None  # No opening bracket found

    end = text.rfind("]")
    if end == -1:
        return None  # No closing bracket found

    json_str = text[start : end + 1]
    try:
        return json.dumps(json.loads(json_str))
    except json.JSONDecodeError:
        return None  # Invalid JSON list

    return None


class DefaultEventExistsClassificationTaskPrompt(Prompt[RamsPassage]):

    def __init__(
        self,
        subject: RamsPassage,
        topic: str,
        answer_value: bool,
        response_interpretter_function=determine_true_false,
    ):
        self.subject = subject
        self.topic = topic
        self.answer_value = answer_value
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:
        return f"""Only respond with either true or false, does the following passage describe one or more events about {self.topic}?\n\n{self.subject.content}\n"""

    def handle_response(self, response: str) -> typing.Optional["Prompt[RamsPassage]"]:
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


def _has_events_of_interest(event_type_prefixes, topic: RamsPassage):

    for event in topic.events:
        for event_type_prefix in event_type_prefixes:
            if event.type_indicator[0].startswith(event_type_prefix):
                return True

    return False


class DefaultEventExistsClassificationTaskPromptFactory(PromptFactory[RamsPassage]):

    def __init__(
        self,
        topic_msg: str,
        event_type_prefixes,
    ):
        self.topic_msg = topic_msg
        self.event_type_prefixes = event_type_prefixes

    def generate_prompt(self, topic: RamsPassage) -> Prompt[RamsPassage]:
        return DefaultEventExistsClassificationTaskPrompt(
            topic,
            self.topic_msg,
            answer_value=_has_events_of_interest(self.event_type_prefixes, topic),
        )

    def create_assessor_func(self):
        return lambda expected_value, actual_value: expected_value == actual_value


class DefaultExtractEventsPrompt(Prompt[RamsPassage]):

    def __init__(
        self,
        subject: RamsPassage,
        topic: str,
        answer_value: bool,
        response_interpretter_function=extract_json_list,
    ):
        self.subject = subject
        self.topic = topic
        self.answer_value = answer_value
        self.answer_response = None
        self.response_interpretter_function = response_interpretter_function

    @property
    def content(self) -> str:
        return f"""Consider the following passage.\n\n{self.subject.content}\n\nIdentify events about {self.topic}. For each event, return a JSON object with an attribute specifying the event type and, if stated, the following event meta-data as additional attributes: the attacker with a "attacker" attribute, the place with a "place" attribute, and the target with "target" attribute. If no relevant events exist, return an empty JSON list. Otherwise return a JSON list with the JSON objects for each event.\n"""

    def handle_response(self, response: str) -> typing.Optional["Prompt[RamsPassage]"]:
        try:
            self.answer_response = self.response_interpretter_function(response)
        except Exception as e:
            print(f"Error handling response in DefaultExtractEventsPrompt: {e}")
        return None


class DefaultExtractEventsPromptFactory(PromptFactory[RamsPassage]):

    def __init__(
        self,
        topic_msg: str,
        event_type_prefixes,
    ):
        self.topic_msg = topic_msg
        self.event_type_prefixes = event_type_prefixes

    def generate_prompt(self, topic: RamsPassage) -> Prompt[RamsPassage]:
        return DefaultExtractEventsPrompt(
            topic,
            self.topic_msg,
            answer_value=_has_events_of_interest(self.event_type_prefixes, topic),
        )

    def create_assessor_func(self):
        return (
            lambda expected_value, actual_value: expected_value
            == non_empty_json_list(actual_value)
        )


def non_empty_json_list(json_str) -> bool:
    if json_str is None:
        return False
    jv = json.loads(json_str)
    if isinstance(jv, list):
        return len(jv) > 0
    return False


def conduct_evaluations(
    tasks: typing.Iterator[RamsPassage],
    prompt_factory: PromptFactory,
    database: db.DatabaseSqlLite,
    d_model: discussion.DiscussionStrategy,
    determine_correct_f=lambda expected_value, actual_value: expected_value
    == actual_value,
):
    for task in tasks:
        # if len(task.synset_options) <= 1:
        #    continue
        try:
            prompt_handler = prompt_factory.generate_prompt(task)
            prompt_strategy_name = prompt_handler.__class__.__name__

            # first check to ensure not already in database
            if database.has_task_result(
                task, prompt_strategy_name, model_id=d_model.model_id
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
                follow_on_prompt = follow_on_prompt.handle_response(
                    follow_on_response[0].content
                )
            end = time.time()
            print(f"Finished discussion ({end-start} seconds).")

            database.add_task_result(
                task,
                log=d.to_json(),
                prompt_strategy=prompt_strategy_name,
                model_id=d_model.model_id,
                answer_value=prompt_handler.answer_value,
                discussion_duration=end - start,
                answer_response=prompt_handler.answer_response,
                correct=determine_correct_f(
                    prompt_handler.answer_value, prompt_handler.answer_response
                ),
            )
        except ValueError as e:
            print(f"Error handling {str(task)}, {str(e)}")


if __name__ == "__main__":
    annotated_passages = read_in_dataset("./data/RAMS_1.0c/data/test.jsonlines")

    event_types_of_interest = {
        "justice.judicialconsequences.extradite",
        "justice.initiatejudicialprocess.n/a",
        "justice.investigate.n/a",
        "justice.judicialconsequences.n/a",
        "justice.initiatejudicialprocess.chargeindict",
        "justice.investigate.investigatecrime",
        "justice.judicialconsequences.execute",
        "justice.initiatejudicialprocess.trialhearing",
        "justice.arrestjaildetain.arrestjaildetain",
    }

    s = set()

    for ap in annotated_passages:
        if (
            len(
                list(
                    e
                    for e in ap.events
                    if e.type_indicator[0].startswith("conflict.attack")
                )
            )
            > 0
        ):
            for e in ap.events:
                s.add(e.type_indicator[0])
        # print(str(ap))

    c = 0
    for ap in annotated_passages:
        if ap.has_event_type(s):
            c += 1
            # print(str(ap))
    print(s)
    print(c)
