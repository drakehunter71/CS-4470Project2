import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from traditionalMachineLearning.traditionalModels import (
    tokenizeSentences,
    wordEmbeddingSentences,
    vectorizeSentences,
)

MLmodels = {
    "rf": "Results/rf_classifier_model_tok.pkl",
    "svm": "Results/svm_classifier_model_tok.pkl",
    "lr": "Results/lr_classifier_model_tok.pkl",
    "xgb": "Results/xgb_classifier_model_tok.pkl",
    "knn": "Results/knn_classifier_model_tok.pkl",
}
ARModels = ["gpt3.5", "gpt4"]

df = pd.read_csv("Data/combined.csv")
df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)
filtered_df = df[df["type"] == "tok"]
train_df = filtered_df[filtered_df["traintest"] == "train"]
test_df = pd.read_csv("Data\speculativeManualClassification.csv")

labelColumns = pd.DataFrame()
labelColumns["spec"] = filtered_df["speculative"].apply(
    lambda x: 1 if x == "Yes" else 0
)
labelColumns["traintest"] = filtered_df["traintest"]

testTokenized = tokenizeSentences(test_df["conclusion2"])
trainTokenized = tokenizeSentences(train_df["data"])
wordEmbeddings = wordEmbeddingSentences(trainTokenized)
testVectorized = vectorizeSentences(testTokenized, wordEmbeddings)
trainVectorized = vectorizeSentences(trainTokenized, wordEmbeddings)

x_test = np.array(testVectorized)
x_train = np.array(trainVectorized)

y_train = labelColumns[labelColumns["traintest"] == "train"]["spec"]
y_test = test_df["speculative"]

for key in MLmodels:
    model_file_path = MLmodels[key]
    with open(model_file_path, "rb") as file:
        model = pickle.load(file)

    train_accuracy = accuracy_score(y_train, model.predict(x_train))
    test_accuracy = accuracy_score(y_test, model.predict(x_test))
    test_f1_score = f1_score(y_test, model.predict(x_test))

    print(f"Training accuracy: {train_accuracy}")
    print(f"Testing accuracy: {test_accuracy}")
    print(f"Testing F1 score: {test_f1_score}")
    print(
        "Classification report:\n",
        classification_report(y_test, model.predict(x_test)),
    )
    print("Confusion matrix:\n", confusion_matrix(y_test, model.predict(x_test)))
    print()

    results_file_name = os.path.join(results_dir, f"{method}_results_{type}.txt")
    model_pkl_file = os.path.join(results_dir, f"{method}_classifier_model_{type}.pkl")

    with open(model_pkl_file, "wb") as file:
        pickle.dump(clf, file)

    results = {"method": "", "f1_score": ""}
    results["method"] = key
    results["f1_score"] = test_f1_score

    result_df = pd.DataFrame([results])

    with open(
        results_file_path,
        "a" if file_exists else "w",
        newline="",
        encoding="utf-8",
    ) as f:
        result_df.to_csv(f, header=not file_exists, index=False)
    file_exists = True

    print(f"Results and model for model '{key}' have been saved.")
