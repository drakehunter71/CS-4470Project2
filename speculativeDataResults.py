import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_csv("Data/combined.csv")
df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)
test_df = df[df["traintest"] == "test"]
autoRegressiveModelResults = pd.read_csv("Results/autoRegressiveModels.csv")

for col in autoRegressiveModelResults.columns():
    f1_score()

traditionalMachineLearningResults = pd.read_csv("Results/traditionalMethodsf1.csv")
