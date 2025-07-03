from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import os

from . import discussion


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
        "model_version": model_version,
    }


class HuggingFaceDiscussionStrategyFactory(discussion.DiscussionStrategyFactory):

    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        pass

    def create(self):
        return HuggingFaceDiscussionStrategy(self.model_name)

    @property
    def model_id(self):
        return f"{self.model_name}"


class HuggingFaceDiscussionStrategy(discussion.DiscussionStrategy):

    def __init__(self, model_name):
        self.model_name = model_name

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for compute
            bnb_4bit_use_double_quant=True,
        )
        # --- Load Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Common practice
        print("Tokenizer loaded.")

        # --- Load Model with device_map and Quantization ---
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,  # Apply the config
            # load_in_8bit=load_in_8bit, # Pass directly if using 8-bit without specific config
            device_map="auto",  # CRITICAL: Distributes model across devices
            torch_dtype=(
                torch.bfloat16 if quantization_config is None else None
            ),  # Use bfloat16 only if not quantizing
            trust_remote_code=False,  # If required by model
            # attn_implementation="flash_attention_1",  # Optional optimization if installed
            # max_memory={0: "10GiB", "cpu": "30GiB"} # Optional: Explicit memory control per device
        )
        print("Model loaded onto devices.")

        # --- STEP 2: Create the Pipeline ---
        # The pipeline will use the already loaded (and potentially quantized/distributed) model.
        # You usually don't need to specify device= here, as device_map handled placement.
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        print("Pipeline created.")
        """
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=os.getenv("HF_API_TOKEN"),
        )"""

    @property
    def model_id(self) -> str:
        return self.model_name

    def speak(self, d, content, role="user"):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that follows directions.",
            },
            {"role": "user", "content": content},
        ]
        outputs = self.pipe(messages, max_new_tokens=256, do_sample=False)
        response_content = outputs[0]["generated_text"][-1]["content"]

        d.messages.append(discussion.Message(role, content))

        response_msg = discussion.Message(
            role="system",
            content=response_content,
        )
        d.messages.append(response_msg)

        return response_msg, response_msg
