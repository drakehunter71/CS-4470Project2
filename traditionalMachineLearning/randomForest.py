import pickle
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
train_df = df[(df["traintest"] == "train") & (df["type"] == "tok")]
test_df = df[(df["traintest"] == "test") & (df["type"] == "tok")]

labels = np.array(
    [
        "Speculative",
        "Not Speculative",
    ],
    dtype="str",
)

inputSentences = train_df["data"].tolist()

labelColumns = pd.DataFrame()
labelColumns["spec"] = df["speculative"].apply(lambda x: 1 if x == "Yes" else 0)
labelColumns["nspec"] = df["speculative"].apply(lambda x: 1 if x == "No" else 0)

inputSentenceTokens = []
for sentence in inputSentences:
    tokens = gensim.utils.simple_preprocess(sentence)
    inputSentenceTokens.append(tokens)

W2V_model_sentence = Word2Vec(
    inputSentenceTokens, min_count=1, vector_size=100, workers=3, window=5, sg=1
)

vectorizedSentences = [None] * len(inputSentenceTokens)
for i in range(len(inputSentenceTokens)):
    sentence = []
    for word in inputSentenceTokens[i]:
        try:
            sentence.append(W2V_model_sentence.wv[word])
        except:
            "do nothing"
    if len(sentence) > 0:
        sentence_avg = np.mean(np.array(sentence, dtype="f"), axis=0)
    else:
        sentence_avg = np.zeros(100)
    vectorizedSentences[i] = sentence_avg

test_size = len(test_df)
train_size = len(train_df)
print(
    "Testing set size: " + str(test_size),
    "|",
    "Training set size: " + str(train_size),
    "|",
    "Total size: " + str(test_size + train_size),
)
# From now on X refers to the document data (title or abstract) and Y refers to the label
# create the X test and training matricies for the article titles.
# Note vectorizedTitles is a list - so format conversion being done
# len(temp) = 20971

temp = np.array(vectorizedSentences)
X_title_test, X_title_train = temp[train_size:], temp[:train_size]


# create the Y test and training arrays for the article labels (list of "np.array columns")
# len(Y_test) = 6 (number of labels)
# len(Y_test[0]) = 4194 (number docs in test set

Y_train, Y_test = [None] * len(labelColumns), [None] * len(labelColumns)
for colNumber in range(len(labelColumns)):
    temp = np.array(labelColumns[colNumber])
    Y_test[colNumber], Y_train[colNumber] = temp[train_size:], temp[:train_size]

# #### Create random forest classifiers - a separate one for each label:
# so exploring 6 binary classifiers here
# sklearn.ensemble has RandomForestClassifier module
# n_estimators, int, default=100,The number of trees in the forest.
# max_depthint, default=None, The maximum depth of the tree. If None,
#       then nodes are expanded until all leaves are pure or until all leaves
#       contain less than min_samples_split samples

print("TITLES:")
title_classifiers = [None] * len(Y_train)
for colNumber in range(len(Y_train)):  # for each of the 6 labels
    temp = RandomForestClassifier(max_depth=6, n_estimators=10)
    temp.fit(X_title_train, Y_train[colNumber])
    title_classifiers[colNumber] = temp
    print(colNumber, labels[colNumber])
    print(
        "Training accuracy:",
        np.sum(temp.predict(X_title_train) == Y_train[colNumber]) / len(X_title_train),
    )
    print(
        "Testing accuracy:",
        np.sum(temp.predict(X_title_test) == Y_test[colNumber]) / len(X_title_test),
    )

    # warnings when there are no positives.
    print(
        "classification report:\n",
        classification_report(temp.predict(X_title_test), Y_test[colNumber]),
    )
    # roc auc gives error when no positives
    # print('roc auc:', roc_auc_score(temp.predict(X_title_test), Y_test[colNumber]))
    print(
        "confusion_matrix:\n",
        confusion_matrix(temp.predict(X_title_test), Y_test[colNumber]),
    )
    print()

print("ABSTRACTS:")
abstract_classifiers = [None] * len(Y_train)
for colNumber in range(len(Y_train)):
    temp = RandomForestClassifier(max_depth=6, n_estimators=10)
    temp.fit(X_abstract_train, Y_train[colNumber])
    abstract_classifiers[colNumber] = temp
    print(colNumber, labels[colNumber])
    print(
        "Training accuracy:",
        np.sum(temp.predict(X_title_train) == Y_train[colNumber]) / len(X_title_train),
    )
    print(
        "Testing accuracy:",
        np.sum(temp.predict(X_title_test) == Y_test[colNumber]) / len(X_title_test),
    )
    # warnings when there are no positives.
    print(
        "classification report:\n",
        classification_report(temp.predict(X_title_test), Y_test[colNumber]),
    )
    # roc auc gives error when no positives
    # print('roc auc:', roc_auc_score(temp.predict(X_title_test), Y_test[colNumber]))
    print(
        "confusion_matrix:\n",
        confusion_matrix(temp.predict(X_title_test), Y_test[colNumber]),
    )
    print()


# #### Create classifier function that evaluates input text on all five labels:
# one for the title, and one for the abstract.
# now defining some utility functions that let you classify new texts


def title_classifier(title):
    global title_classifiers
    tokenTitle = gensim.utils.simple_preprocess(title)
    vecTitle = []
    for word in tokenTitle:
        try:
            vecTitle.append(W2V_model_title.wv[word])
        except:
            "do nothing"
    vecTitle = np.mean(np.array(vecTitle, dtype="f"), axis=0)
    preds = [None] * len(title_classifiers)  # the 6 title based classifiers
    for index in range(len(title_classifiers)):
        # the reshape makes it a 2 D array from the 1 D array of vecTitle
        # len(vecTitle)
        # len(vecTitle.reshape(1, -1))
        # len(vecTitle.reshape(1, -1))[0]

        preds[index] = int(title_classifiers[index].predict(vecTitle.reshape(1, -1))[0])
    return np.array(preds)


def abstract_classifier(abstract):
    global abstract_classifiers
    tokenAbstract = gensim.utils.simple_preprocess(abstract)
    vecAbstract = []
    for word in tokenAbstract:
        try:
            vecAbstract.append(W2V_model_abstract.wv[word])
        except:
            "do nothing"
    vecAbstract = np.mean(np.array(vecAbstract, dtype="f"), axis=0)
    preds = [None] * len(abstract_classifiers)
    for index in range(len(abstract_classifiers)):
        preds[index] = int(
            abstract_classifiers[index].predict(vecAbstract.reshape(1, -1))[0]
        )
    return np.array(preds)


# #### Try out classifier on some made up article name inputs:

# title classifier

articleName = "New Methods for KNN with text data"
preds = title_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

articleName = "Pi used in new formula"
preds = title_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

articleName = "New prime number discovered"
preds = title_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

articleName = "New Data distribution used to speed up training"
preds = title_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

# abstract classifier

articleName = "New Methods for KNN with text data"
preds = abstract_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

articleName = "Pi used in new formula"
preds = abstract_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

articleName = "New prime number discovered"
preds = abstract_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])

articleName = "New Data distribution used to speed up training"
preds = abstract_classifier(articleName)
print("Output vector:", preds, "|", "Predicted Label(s):", labels[preds == 1])


# model_pkl_file = "rf_classifier_model.pkl"

# with open(model_pkl_file, "wb") as file:
#     pickle.dump(model, file)
