from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig, AutoConfig

from . import discussion

from huggingface_hub import HfApi




class Llama2_13BDiscussionStrategyFactory:

    def __init__(self, model_name="TheBloke/Llama-2-13B-Chat-GPTQ"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        config = AutoConfig.from_pretrained(model_name)
        config.loss_type = "ForCausalLMLoss"  # Explicitly set loss_type
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                  config=config,  # Pass the updated config
                                                  device_map='auto', 
                                                  trust_remote_code=False, 
                                                  revision='main')
        self.model_name = model_name
        pass

    def create(self):
        return Llama2_13BDiscussionStrategy(self.model, self.tokenizer)
    
    # def get_model_info(self):
    #     api = HfApi()
    #     model_info = api.model_info(self.model_name)
        
    #     model_name = model_info.id.split("/")[-1]
    #     model_size = ""  # Extract size if available in model tags
    #     model_version = model_info.sha  # Model's latest commit SHA
        
    #     # Extracting model size if present in the model tags
    #     for tag in model_info.tags:
    #         if "B" in tag:  # e.g., "13B"
    #             model_size = tag
    #             break
        
    #     return {
    #         "model_id": model_info.id,
    #         "model_name": model_name,
    #         "model_size": model_size,
    #         "model_version": model_version
    #     }


    @property
    def model_id(self):
        return f"{self.model_name}"


class Llama2_13BDiscussionStrategy:

    def __init__(self, model, tokenizer):
        self.discussion = discussion.Discussion()
        self.model = model
        self.tokenizer = tokenizer


    def speak(self, content, role="user"):

        self.discussion.messages.append(discussion.Message(role,content))

        # Define generation configuration
        generation_config = GenerationConfig(
        temperature=0.5,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512
        )

        if "True or false" in content:
            prompt_template = f'''[INST] {content}
            ### ANSWER: [/INST]
            '''
            input_ids = self.tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            outputs = self.model.generate(inputs=input_ids, generation_config=generation_config)
            # response_content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_content = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split("\n")[-1].strip()
        else: 
            prompt_template = f'''[INST] <<SYS>>
            You are an AI assistant that responds with a valid JSON list. 
            Do not include any additional text, explanations, or formatting outside the JSON structure.
            Strictly follow this format:

            Example:
            [{{"name": "Alice", "age": 25, "city": "New York"}},
            {{"name": "Bob", "age": 30, "city": "Los Angeles"}}]

            Now, generate a JSON list for the following query:<</SYS>>
            {content}
            ### ANSWER: [/INST]
            '''
        

            input_ids = self.tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            outputs = self.model.generate(inputs=input_ids, generation_config=generation_config)

            # Convert Tensor to list before decoding
            response_content = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip().split("[/INST]")[-1].strip()

        response_msg = discussion.Message(
            role="system",
            content=response_content,
        )        
        self.discussion.messages.append(response_msg)

        return response_msg, response_msg

