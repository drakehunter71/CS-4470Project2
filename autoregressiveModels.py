import pandas as pd
import time
from openaiapi import OpenAiManager
from anthropicapi import AnthropicManager

df = pd.read_csv("Data/combined.csv")
test_df = df[df["traintest"] == "test"]


openai_manager = OpenAiManager()
openai_model = {3.5: "gpt-3.5-turbo", 4: "gpt-4-turbo-preview"}
anthropic_manager = AnthropicManager()
anthropic_model = {
    "Haiku": "claude-3-haiku-20240307",
    "Sonnet": "claude-3-sonnet-20240229",
    "Opus": "claude-3-opus-20240229",
}

print(
    df[
        df["traintest"] == "train"
        and df["type"] == "tok"
        and df["speculative"] == "Yes"
    ].head(1)
)
print(
    df[
        df["traintest"] == "train"
        and df["type"] == "stm"
        and df["speculative"] == "Yes"
    ].head(1)
)
print(
    df[
        df["traintest"] == "train"
        and df["type"] == "bgm"
        and df["speculative"] == "Yes"
    ].head(1)
)
print(
    df[
        df["traintest"] == "train" and df["type"] == "tok" and df["speculative"] == "No"
    ].head(1)
)
print(
    df[
        df["traintest"] == "train" and df["type"] == "stm" and df["speculative"] == "No"
    ].head(1)
)
print(
    df[
        df["traintest"] == "train" and df["type"] == "bgm" and df["speculative"] == "No"
    ].head(1)
)


examples = False
if examples:
    initial_message1 = """I am going to give you a set of words and I want you to decide if the set of words is speculative or not. 
    The set of words could be tokenized, tokenized and stemmed, or tokenized, stemmed, and bi-grammed.
    Respond in only 1 word, 'Yes' or 'No', if the following sentence is speculative.
    """
    ex = """
    Tokenized Speculative Example:
    Tokenized and Stemmed Speculative Example:
    Tokenized, Stemmed, and Bi-grammed Speculative Example:
    Tokenized Non-speculative Example:
    Tokenized and Stemmed Non-speculative Example:
    Tokenized, Stemmed, and Bi-grammed Non-speculative Example:
    """

    initial_message2 = """Remember, If the set of words is speculative, respond 'Yes' and if the set of words is not speculative, respond 'No."
    The message is: 
    """
    initial_message = initial_message1 + ex + initial_message2
    # ADD EXAMPLES AND REFORMAT MESSAGE
else:
    initial_message = """I am going to give you a set of words and I want you to decide if the set of words is speculative or not. 
    The set of words could be tokenized, tokenized and stemmed, or tokenized, stemmed, and bi-grammed.
    Respond in only 1 word, 'Yes' or 'No', if the following sentence is speculative.
    Remember, If the set of words is speculative, respond 'Yes' and if the set of words is not speculative, respond 'No."
    The message is: 
    """
exit()
gpt3_5_results = []
gpt4_results = []
haiku_results = []
sonnet_results = []
opus_results = []

test_df = test_df.head(10)

for data in test_df["data"]:
    data = data.rstrip(" .")
    for model in openai_model:
        # openai_result = model
        openai_result = openai_manager.chat(
            prompt=initial_message + data, model_name=openai_model[model]
        )
        if model == 3.5:
            gpt3_5_results.append(openai_result)
        else:
            gpt4_results.append(openai_result)
    time.sleep(1.3)  # anthropic rate limit is 50 RPM
    for model in anthropic_model:
        # anthropic_result = model
        anthropic_result = anthropic_manager.chat(
            prompt=initial_message + data, model_name=anthropic_model[model]
        )
        if model == "Haiku":
            haiku_results.append(anthropic_result)
        elif model == "Sonnet":
            sonnet_results.append(anthropic_result)
        else:
            opus_results.append(anthropic_result)
data = {
    "GPT3.5": gpt3_5_results,
    "GPT4": gpt4_results,
    "Haiku": haiku_results,
    "Sonnet": sonnet_results,
    "Opus": opus_results,
}
df = pd.DataFrame(data)

if examples:
    print("")
    df.to_csv("Results/autoregressiveModelsExamples.csv", index=False)
else:
    df.to_csv("Results/autoregressiveModels.csv", index=False)
