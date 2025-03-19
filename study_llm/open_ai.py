import requests
import time
import typing
import json
import os

from . import discussion


API_URL = "https://api.openai.com/v1/chat/completions"

session = requests.Session()
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}
session.headers.update(headers)

class OpenAIStrategyFactory(discussion.DiscussionStrategyFactory):

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        pass

    def create(self):
        return OpenAiDiscussionStrategy(self.model_name)

    @property
    def model_id(self):
        return f"{self.model_name}"


def send_request(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000
    }
    
    response = session.post(API_URL, data=json.dumps(payload))
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    return response.json()


# Function to parse response
def parse_response(response):
    try:
        # Try parsing as a boolean true/false response
        content = response['choices'][0]['message']['content']
        s_content = content.strip().lower()
        if s_content.starts_with("true"):
            return "true"
        elif s_content.starts_with("false"):
            return "false"
        else:
            # Try parsing as JSON response
            return json.loads(content)
    except json.JSONDecodeError:
        return content
    except KeyError as e:
        raise Exception(f"Unexpected response format: {e}")


class OpenAiDiscussionStrategy(discussion.DiscussionStrategy):

    def __init__(self, model_name):
        self.model_name = model_name
        
    @property
    def model_id(self) -> str:
        return self.model_name    

    def speak(self, d, content, role="user", max_tokens=4096, temperature=0):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that follows directions."},
            {"role": "user", "content": content},
        ]
        d.messages.append(discussion.Message(role,content))

        # Make an API request to OpenAI's language model
        start = time.time()
        #messages = [
        #    {"role": "system", "content": "Summarize the sentiment expressed in the user content. Reason about the entity that caused the sentiment, how the entity caused the sentiment, the topic affected by the sentiment, and the contextual details. Extract the content as JSON."},
        #    {"role": "user", "content": prompt},
        #]

        response = send_request(messages, model=self.model_id)

        duration = time.time() - start

        #parsed_response = parse_response(response)

        '''
        try:
            sr = completion.choices[0].message.parsed

            #response_text = response.choices[0].text.strip()
            #response_json = json.loads(response_text)  # Parse the response as JSON
            print("Valid JSON received:")
            print(json.dumps(sr.dict(), indent=4))  # Print formatted JSON
            messages.append(completion.choices[0].message.dict())

            return SentimentSummary(
                content_hash=content.content_hash,
                model_id=model,
                prompt_strategy=prompt_strategy,
                sentiment=True if sr.sentiment == 'positive' else (False if sr.sentiment == 'negative' else None),
                justifications=sr.justifications_of_sentiment,
                topic=sr.topic,
                topic_lemma=sr.topic_lemma,
                location=sr.lat_lng_of_topic_location,
                content_datetime=sr.datetime_of_topic,
                method=sr.actions_causing_sentiment,
                method_lemma=sr.action_lemma,
                contributors=sr.names_of_contributors_that_cause_sentiment,
                discussion_duration=duration,
                log=messages,
                topic_values=None,
                contributors_values=None,
                method_values=None,
            )
        except json.JSONDecodeError as e:
            print(f"The response was not valid JSON. Here is the raw error: {e}")
            return None
    
        '''

        response_content = response['choices'][0]['message']['content']

        response_msg = discussion.Message(
            role="system",
            content=response_content,
        )        
        d.messages.append(response_msg)

        return response_msg, response_msg

