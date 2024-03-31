import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openaiapi import OpenAiManager
from anthropicapi import AnthropicManager

# 3.5 vs. 4
# Haiku vs. Sonnet vs. Opus
# giving example sentences vs. not giving example sentences
basefilestring = "C:/DrakeSmith/Visual Studio Projects/HealthDataAnalytics/CS-4470Project2/Data/speculative/Speculation-Dataset/"
nspec_test_file_list = [
    "tok/nspec_test.tok",
    "stm/nspec_test.stm",
    "bgm/nspec_test.bgm",
]
nspec_train_file_list = [
    "tok/nspec_train.tok",
    "stm/nspec_train.stm",
    "bgm/nspec_train.bgm",
]
spec_test_file_list = ["tok/spec_test.tok", "stm/spec_test.stm", "bgm/spec_test.bgm"]
spec_train_file_list = [
    "tok/spec_train.tok",
    "stm/spec_train.stm",
    "bgm/spec_train.bgm",
]

df = pd.read_csv(basefilestring + nspec_test_file_list[0], header=None)
print(df)

openai_manager = OpenAiManager()
openai_model = {3.5: "gpt-3.5-turbo", 4: "gpt-4-turbo-preview"}
anthropic_manager = AnthropicManager()
anthropic_model = {
    "Haiku": "claude-3-haiku-20240307",
    "Sonnet": "claude-3-sonnet-20240229",
    "Opus": "claude-3-opus-20240229",
}

initial_message = """I am going to give you a set of words and I want you to decide if the set of words is speculative or not. 
The set of words could be tokenized, tokenized and stemmed, or tokenized, stemmed, and bi-grammed.
Respond in only 1 word, 'Yes' or 'No', if the following sentence is speculative.
Remember, If the set of words is speculative, respond 'Yes' and if the set of words is not speculative, respond 'No."
The message is: 
"""
sentence = "Key components of the programmed cell death pathway are conserved between Caenorhabditis elegans Drosophila melanogaster and humans  ."
sentence = sentence.rstrip(" .")

openai_result = openai_manager.chat(
    prompt=initial_message + sentence, model_name=openai_model[3.5]
)
anthropic_result = anthropic_manager.chat(
    prompt=initial_message + sentence, model_name=anthropic_model["Haiku"]
)

# df = pd.DataFrame(columns=["sentence", "speculative", "gptanswer"])
# df = df._append(
#     {"sentence": sentence, "speculative": "Yes", "gptanswer": "No"}, ignore_index=True
# )
# print(df)
# df.to_csv("Results/test.csv")
