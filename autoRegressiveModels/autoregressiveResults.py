import pandas as pd
import os
from sklearn.metrics import f1_score

df = pd.read_csv("Data/combined.csv")
df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)
test_df = df[df["traintest"] == "test"].copy()
test_df["speculative"] = (
    test_df["speculative"]
    .str.lower()
    .str.replace(r"[.\s]+$", "", regex=True)
    .map({"yes": 1, "no": 0})
    .astype(int)
)
test_df.reset_index(drop=True, inplace=True)

autoRegressiveModelResults = pd.read_csv("Results/autoRegressiveModelsUpdated.csv")

autoRegressiveModelResults.loc[:, "type"] = test_df.loc[:, "type"]
autoRegressiveModelResults.loc[:, "speculative"] = test_df.loc[:, "speculative"]

models = ["GPT3.5", "GPT4", "Haiku", "Sonnet", "Opus"]
types = ["tok", "stm", "bgm"]

results_file_path = "Results/autoRegressiveMethodsf1.csv"
file_exists = os.path.isfile(results_file_path)

for model in models:
    autoRegressiveModelResults[model] = (
        autoRegressiveModelResults[model]
        .str.lower()
        .str.replace(r"[.\s]+$", "", regex=True)
        .map({"yes": 1, "no": 0})
        .astype(int)
    )

    for type in types:
        filtered_df = autoRegressiveModelResults[
            autoRegressiveModelResults["type"] == type
        ]

        y_true = filtered_df["speculative"]
        y_pred = filtered_df[model]

        test_f1_score = f1_score(y_true, y_pred)

        results = {"method": model, "type": type, "f1_score": test_f1_score}

        result_df = pd.DataFrame([results])
        with open(
            results_file_path, "a" if file_exists else "w", newline="", encoding="utf-8"
        ) as f:
            result_df.to_csv(f, header=not file_exists, index=False)
        file_exists = True
