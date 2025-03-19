import study_llm.event_extraction_task.prompts as prompts
from dotenv import load_dotenv


load_dotenv()  # load environment variables from .env.
# to use huggingface models (loaded from their website) set HF_API_TOKEN in file "".env"

datasets = [
    "data/maven/train.jsonl"
]

event_types = [
    "Violence", "Surrounding", "Besieging", "Attack", "Military_operation", "Hostile_encounter", "Terrorism", "Bearing_arms", "Defending", "Killing"
]

topic_msg = ", ".join(e.lower().replace("_", " ") for e in event_types[:-1]) + ", or " + event_types[-1].lower()


# Prepare the creator(s) of the prompts
p_factories = [
    prompts.DefaultEventExistsClassificationTaskPromptFactory(topic_msg),
    prompts.DefaultExtractEventsPromptFactory(topic_msg),
]

huggingface_models = [
    #"meta-llama/Llama-3.2-3B-Instruct",
    #"mistralai/Mistral-7B-Instruct-v0.3",
    #"meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
]