import typing
import json
from huggingface_hub import HfApi

def get_model_info(model_name):
        api = HfApi()
        model_info = api.model_info(model_name)
        
        model_name = model_info.id.split("/")[-1]
        model_size = ""  # Extract size if available in model tags
        model_version = model_info.sha  # Model's latest commit SHA
        
        # Extracting model size if present in the model tags
        for tag in model_info.tags:
            if "B" in tag:  # e.g., "13B"
                model_size = tag
                break
        
        return {
            "model_id": model_info.id,
            "model_name": model_name,
            "model_size": model_size,
            "model_version": model_version
        }


class SynsetOption:

    def __init__(self, id:str, gloss:str):
        self.id = id
        self.gloss = gloss


class EventExtractionTask:

    def __init__(self,
                 sentence,
                 word,
                 synset_answer,
                 synset_options:typing.List[SynsetOption],
                 pos = "NOUN",
                 text_id = None,
                 text_order_id = None,
                 ):
        self.sentence = sentence
        self.word = word
        self.synset_answer = synset_answer
        self.synset_options = synset_options
        self.pos = pos
        self.text_id = text_id
        self.text_order_id = text_order_id

        found = False
        for opt in synset_options:
            if opt.id == self.synset_answer:
                assert not found # ensure only one option matches answer
                found = True

        if not found:
            print(f"Did not find {self.synset_answer} in {[o.id for o in synset_options]}!")
            #assert found    

    
    def to_json(self):
        return {
            "sentence":self.sentence,
            "word":self.word,
            "answer":self.synset_answer,
            "options":list({"id":o.id, "gloss":o.gloss } for o in self.synset_options),
        }
    
    def to_json_complete(self):
        return {
            "sentence":self.sentence,
            "word":self.word,
            "answer":self.synset_answer,
            "pos":self.pos,
            "options":list({"id":o.id, "gloss":o.gloss } for o in self.synset_options),
            "text_id":self.text_id,
            "text_order_id":self.text_order_id,
        }
    
    @staticmethod
    def From_json(j):
        pos = "NOUN" if "pos" not in j else j["pos"]
        return EventExtractionTask(j["sentence"],j["word"], j["answer"], [SynsetOption(o["id"], o["gloss"]) for o in j["options"]], pos)

    @staticmethod
    def Load_from_json(file_path:str) -> typing.List["EventExtractionTask"]:
        evaluations = []
        with open(file_path, "r", encoding="cp1252", errors="ignore") as f:
            obj = []
            for line in f:
                obj.append(json.load(line.strip))

            for wse_obj in obj:
                wse = EventExtractionTask(
                    sentence=wse_obj["sentence"],
                    word=wse_obj["word"],
                    synset_answer=wse_obj["answer"],
                    pos=wse_obj["pos"],
                    synset_options=list(SynsetOption(o["id"], o["gloss"]) for o in wse_obj["options"]),
                    text_id=wse_obj["text_id"],
                    text_order_id=wse_obj["text_order_id"],
                )
                evaluations.append(wse)

        return evaluations
