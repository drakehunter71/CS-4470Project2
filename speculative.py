import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from chatgpt import OpenAiManager

# Text classifiers

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

openai_manager = OpenAiManager()
initial_message = """I am going to give you a set of words and I want you to decide if the set of words is speculative or not. 
The set of words could be tokenized, tokenized and stemmed, or tokenized, stemmed, and bi-grammed.
Respond in only 1 word, 'Yes' or 'No', if the following sentence is speculative.
Remember, If the set of words is speculative, respond 'Yes' and if the set of words is not speculative, respond 'No."
The message is: 
"""
sentence = "Key components of the programmed cell death pathway are conserved between Caenorhabditis elegans Drosophila melanogaster and humans  ."
sentence = sentence.rstrip(" .")
print(initial_message + sentence)
openai_result = openai_manager.chat(initial_message + sentence)
