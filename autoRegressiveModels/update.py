import pandas as pd
import time
import logging
import json
import os
from openaiapi import OpenAiManager
from anthropicapi import AnthropicManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

initial_message = """I am going to give you a set of words and I want you to decide if the set of words is speculative or not. 
    The set of words could be tokenized, tokenized and stemmed, or tokenized, stemmed, and bi-grammed.
    Respond in only 1 word, 'Yes' or 'No', if the following sentence is speculative.
    Remember, If the set of words is speculative, respond 'Yes' and if the set of words is not speculative, respond 'No."
    The set of words is: 
    """


def load_model_config(path="autoRegressiveModels\model_config.json"):
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


def update_errors_in_results(results_file_path):
    config = load_model_config()
    if not config:
        return
    df = pd.read_csv("Data/combined.csv")
    df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)
    test_df = df[df["traintest"] == "test"]

    if not os.path.exists(results_file_path):
        print("Results file does not exist.")
        return

    df = pd.read_csv(results_file_path)

    openai_manager = OpenAiManager()
    anthropic_manager = AnthropicManager()

    for index, row in df.iterrows():
        for column in df.columns:
            if row[column] == "ERROR":
                if "GPT" in column:
                    manager = openai_manager
                    model_name = config["openai_models"][column]
                else:
                    manager = anthropic_manager
                    model_name = config["anthropic_models"][column]
                updated_result = make_api_call(
                    manager,
                    initial_message + test_df.iloc[[index]]["data"].item(),
                    model_name,
                )
                time.sleep(1.3)
                df.at[index, column] = updated_result
    df.to_csv("Results/autoregressiveModelsUpdated.csv", index=False)
    print("Results file has been updated.")


results_file_path = "Results/autoregressiveModels.csv"
update_errors_in_results(results_file_path)
