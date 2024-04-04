import pickle
import os
import pandas as pd
import numpy as np
import gensim.utils
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score,
    confusion_matrix,
    mean_squared_log_error,
    r2_score,
    f1_score,
)

df = pd.read_csv("Data/combined.csv")
df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)

results_dir = "Results"
os.makedirs(results_dir, exist_ok=True)

types = ["tok", "stm", "bgm"]


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
    clf = RandomForestClassifier(max_depth=6, n_estimators=10)
    clf.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, clf.predict(x_train))
    test_accuracy = accuracy_score(y_test, clf.predict(x_test))

    print(f"Training accuracy: {train_accuracy}")
    print(f"Testing accuracy: {test_accuracy}")
    print(
        "Classification report:\n",
        classification_report(y_test, clf.predict(x_test)),
    )
    print("Confusion matrix:\n", confusion_matrix(y_test, clf.predict(x_test)))
    print()

    results_file_name = os.path.join(results_dir, f"results_{type}.txt")
    model_pkl_file = os.path.join(results_dir, f"rf_classifier_model_{type}.pkl")

    with open(results_file_name, "w") as results_file:
        results_file.write(f"Type: {type}\n")
        results_file.write(f"Training accuracy: {train_accuracy}\n")
        results_file.write(f"Testing accuracy: {test_accuracy}\n")
        results_file.write("Classification report:\n")
        results_file.write(classification_report(y_test, clf.predict(x_test)))
        results_file.write("Confusion matrix:\n")
        results_file.write(str(confusion_matrix(y_test, clf.predict(x_test))) + "\n")

    with open(model_pkl_file, "wb") as file:
        pickle.dump(clf, file)

    print(f"Results and model for type '{type}' have been saved.")
