import pandas as pd
import numpy as np
import gensim.utils
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier
import pickle
import os

methods = ["rf", "svm", "lr", "xgb", "knn"]
types = ["tok", "stm", "bgm"]


def get_model(method):
    if method == "rf":
        return RandomForestClassifier(max_depth=6, n_estimators=10)
    elif method == "svm":
        return SVC(kernel="linear")
    elif method == "lr":
        return LogisticRegression(max_iter=1000)
    elif method == "xgb":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif method == "knn":
        return KNeighborsClassifier()
    else:
        raise ValueError(f"Method {method} not recognized.")


df = pd.read_csv("Data/combined.csv")
df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)

results_dir = "Results"
os.makedirs(results_dir, exist_ok=True)
results_file_path = "Results/traditionalMethodsf1.csv"
file_exists = os.path.isfile(results_file_path)


def tokenizeSentences(inputSentences):
    inputSentenceTokens = []
    for sentence in inputSentences:
        tokens = gensim.utils.simple_preprocess(sentence)
        inputSentenceTokens.append(tokens)
    return inputSentenceTokens


def wordEmbeddingSentences(inputSentenceTokens):
    W2V_model_sentence = Word2Vec(
        inputSentenceTokens, min_count=1, vector_size=100, workers=3, window=5, sg=1
    )
    return W2V_model_sentence


def vectorizeSentences(inputSentenceTokens, W2V_model_sentence):
    vectorizedSentences = [None] * len(inputSentenceTokens)
    for i in range(len(inputSentenceTokens)):
        sentence = []
        for word in inputSentenceTokens[i]:
            try:
                sentence.append(W2V_model_sentence.wv[word])
            except:
                pass
        if len(sentence) > 0:
            sentence_avg = np.mean(np.array(sentence, dtype="f"), axis=0)
        else:
            sentence_avg = np.zeros(100)
        vectorizedSentences[i] = sentence_avg
    return vectorizedSentences


for type in types:
    break
    filtered_df = df[df["type"] == type]
    train_df = filtered_df[filtered_df["traintest"] == "train"]
    test_df = filtered_df[filtered_df["traintest"] == "test"]

    inputSentences = train_df["data"].tolist()

    labelColumns = pd.DataFrame()
    labelColumns["spec"] = filtered_df["speculative"].apply(
        lambda x: 1 if x == "Yes" else 0
    )
    labelColumns["traintest"] = filtered_df["traintest"]

    test_size = len(test_df)
    train_size = len(train_df)

    print(
        "Testing set size: " + str(test_size),
        "|",
        "Training set size: " + str(train_size),
        "|",
        "Total size: " + str(test_size + train_size),
    )

    testTokenized = tokenizeSentences(test_df["data"])
    trainTokenized = tokenizeSentences(train_df["data"])
    wordEmbeddings = wordEmbeddingSentences(trainTokenized)
    testVectorized = vectorizeSentences(testTokenized, wordEmbeddings)
    trainVectorized = vectorizeSentences(trainTokenized, wordEmbeddings)

    x_test = np.array(testVectorized)
    x_train = np.array(trainVectorized)

    y_train = labelColumns[labelColumns["traintest"] == "train"]["spec"]
    y_test = labelColumns[labelColumns["traintest"] == "test"]["spec"]

    # Predicting if Sentence is Speculative
    print("")
    print("Type: " + type)
    for method in methods:
        print("Method: " + method)
        clf = clf = get_model(method)
        clf.fit(x_train, y_train)

        train_accuracy = accuracy_score(y_train, clf.predict(x_train))
        test_accuracy = accuracy_score(y_test, clf.predict(x_test))
        test_f1_score = f1_score(y_test, clf.predict(x_test))

        print(f"Training accuracy: {train_accuracy}")
        print(f"Testing accuracy: {test_accuracy}")
        print(f"Testing F1 score: {test_f1_score}")
        print(
            "Classification report:\n",
            classification_report(y_test, clf.predict(x_test)),
        )
        print("Confusion matrix:\n", confusion_matrix(y_test, clf.predict(x_test)))
        print()

        results_file_name = os.path.join(results_dir, f"{method}_results_{type}.txt")
        model_pkl_file = os.path.join(
            results_dir, f"{method}_classifier_model_{type}.pkl"
        )

        with open(results_file_name, "w") as results_file:
            results_file.write(f"Type: {type}\n")
            results_file.write(f"Training accuracy: {train_accuracy}\n")
            results_file.write(f"Testing accuracy: {test_accuracy}\n")
            results_file.write(f"Testing F1 score: {test_f1_score}")
            results_file.write("Classification report:\n")
            results_file.write(classification_report(y_test, clf.predict(x_test)))
            results_file.write("Confusion matrix:\n")
            results_file.write(
                str(confusion_matrix(y_test, clf.predict(x_test))) + "\n"
            )

        with open(model_pkl_file, "wb") as file:
            pickle.dump(clf, file)

        results = {"method": "", "type": "", "f1_score": ""}
        results["method"] = method
        results["type"] = type
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

        print(f"Results and model for type '{type}' have been saved.")
