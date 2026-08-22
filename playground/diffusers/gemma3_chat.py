# Gemma-3 4B Chat pipeline
#
# requires 4-6GB VRAM

import os

# Disable XET transfer because my home router keeps freezing
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_EMERGENCY_RETRY"] = "1"

import locale
import sys

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from transformers import GenerationConfig, TextStreamer, pipeline, utils

utils.logging.set_verbosity_error()


SYSTEM_PROMPT = f"""
You are an authentic, adaptive AI collaborator and a knowledgeable peer.
Your goal is to address the user's true intent with insightful, yet clear and concise responses.
Your tone must be warm, and approachable.
Actively balance empathy with candor: validate the user's feelings, efforts,
or frustrations, and explain concepts clearly without ever sounding like a formal, pedantic, or rigid lecturer.

You must reply using the user's language({locale.getlocale()}),
"""


def load_pipe(model_id: str = "unsloth/gemma-3-4b-it-qat-bnb-4bit"):
    pipe = pipeline("text-generation", model=model_id, clean_up_tokenization_spaces=False, device_map="auto")
    return pipe


def main():
    pipe = load_pipe()

    generation_config = GenerationConfig(
        max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=pipe.tokenizer.eos_token_id
    )
    streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
    memory_history = InMemoryHistory()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = prompt("You > ", history=memory_history).strip()
        if user_input.startswith("/q") or user_input == "quit":
            sys.exit(0)
        if not user_input:
            continue

        print("AI : ", end="", flush=True)
        messages.append({"role": "user", "content": user_input})
        prompt_text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt_text, generation_config=generation_config, streamer=streamer)
        generated_text = outputs[0]["generated_text"]
        ai_response = generated_text[len(prompt_text) :].strip()

        messages.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    main()
