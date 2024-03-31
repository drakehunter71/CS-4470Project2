import pandas as pd
import time
import logging
import json
import os
from openaiapi import OpenAiManager
from anthropicapi import AnthropicManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_model_config(path="model_config.json"):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Failed to load model configuration: {e}")
        return None


def make_api_call(manager, prompt, model_name):
    try:
        return manager.chat(prompt=prompt, model_name=model_name)
    except Exception as e:
        logging.error(f"API call failed for model {model_name}: {e}")
        return "ERROR"


def main():
    config = load_model_config()
    if not config:
        return

    df = pd.read_csv("Data/combined.csv")
    test_df = df[df["traintest"] == "test"]

    openai_manager = OpenAiManager()
    anthropic_manager = AnthropicManager()

    initial_message = """I am going to give you a set of words and I want you to decide if the set of words is speculative or not. 
    The set of words could be tokenized, tokenized and stemmed, or tokenized, stemmed, and bi-grammed.
    Respond in only 1 word, 'Yes' or 'No', if the following sentence is speculative.
    Remember, If the set of words is speculative, respond 'Yes' and if the set of words is not speculative, respond 'No."
    The set of words is: 
    """

    results_file_path = "Results/autoregressiveModels.csv"
    file_exists = os.path.isfile(results_file_path)

    for counter, data in enumerate(test_df["data"], start=1):
        results = {"GPT3.5": "", "GPT4": "", "Haiku": "", "Sonnet": "", "Opus": ""}

        data_prompt = initial_message + data.rstrip(" .")

        for model_name, model_id in config["openai_models"].items():
            result = make_api_call(openai_manager, data_prompt, model_id)
            results[model_name] = result

        time.sleep(1.3)

        for model_name, model_id in config["anthropic_models"].items():
            result = make_api_call(anthropic_manager, data_prompt, model_id)
            results[model_name] = result

        result_df = pd.DataFrame([results])

        with open(
            results_file_path, "a" if file_exists else "w", newline="", encoding="utf-8"
        ) as f:
            result_df.to_csv(f, header=not file_exists, index=False)
        file_exists = True

        if counter % 100 == 0 or counter == len(test_df):
            logging.info(f"Processed {counter} entries.")


if __name__ == "__main__":
    main()
